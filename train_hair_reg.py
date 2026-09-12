#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from omegaconf import OmegaConf
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, merge_render
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from utils.sh_utils import eval_sh
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, head_iteration):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # frozen, pre-trained FLAME head
    gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
    mesh_renderer = NVDiffRenderer()
    scene = Scene(dataset, gaussians, load_iteration=head_iteration)

    name = dataset.scale_gaus.split('/')[-1]
    knn_path = dataset.scale_gaus.replace(name, 'knn_indices.npy')
    knn = torch.from_numpy(np.load(knn_path)).cuda()

    opt.is_hair_opt = True
    gaussians.training_setup(opt)  # freezes the head (dummy no-op param)

    # trainable hair Gaussians, initialized from the Frenet-frame fit
    gaussians_hair = GaussianModel(dataset.sh_degree)
    scenehair = Scene(dataset, gaussians_hair)
    gaussians_hair.training_setup(opt)  # only f_dc/f_rest are optimized

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians_hair.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image = None
                custom_cam, msg = network_gui.receive()
                if custom_cam != None:
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(int(dataset.fix_tstep), msg['use_original_mesh'])
                    if msg['show_splatting']:
                        net_image = merge_render(custom_cam, gaussians, gaussians_hair, pipe, background, msg['scaling_modifier'])["render"]
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)
                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]
                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians_hair._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians_hair.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            gaussians_hair.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
            bg = viewpoint_cam.bg
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)
            bg = viewpoint_cam.bg

        # head geometry is frozen at a single fixed timestep
        gaussians.select_mesh_by_timestep(int(dataset.fix_tstep))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        render_pkg = merge_render(viewpoint_cam, gaussians, gaussians_hair, pipe, background)
        image, radii, info = render_pkg["render"], render_pkg["radii"], render_pkg["info"]

        info["means2d"].retain_grad()
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        visibility_filter = radii > 0

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        hair_mask = viewpoint_cam.hair_mask.cuda()

        if iteration in saving_iterations:
            wandb.log({
                "train_preview/render": wandb.Image(image.clamp(0.0, 1.0)),
                "train_preview/gt": wandb.Image(gt_image.clamp(0.0, 1.0)),
                "train_preview/hair_masked": wandb.Image((image * hair_mask).clamp(0.0, 1.0)),
            }, step=iteration)

        losses = {}
        losses['l1'] = l1_loss(image * hair_mask, gt_image * hair_mask) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image * hair_mask, gt_image * hair_mask)) * opt.lambda_dssim

        if iteration > 3000:
            shs_view = gaussians_hair.get_features.transpose(1, 2).view(-1, 3, (gaussians_hair.max_sh_degree + 1) ** 2)
            dir_pp = gaussians_hair.get_xyz - viewpoint_cam.camera_center.cuda().repeat(gaussians_hair.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians_hair.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp_hair = torch.clamp_min(sh2rgb + 0.5, 0.0)

            n_strands = knn.shape[0]
            colors_precomp_hair = colors_precomp_hair.reshape(n_strands, -1, 3)
            neighbor_colors = colors_precomp_hair[knn]
            mean_neighbor_colors = neighbor_colors.mean(dim=1)
            losses['color'] = F.mse_loss(colors_precomp_hair, mean_neighbor_colors)

        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'color' in losses:
                    postfix["color"] = f"{losses['color']:.{7}f}"
                postfix["#points"] = gaussians_hair._xyz.shape[0]
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians, gaussians_hair, dataset, pipe, background)
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scenehair.save(iteration)

            # Optimizer step — only the hair model trains, head stays frozen
            if iteration < opt.iterations:
                gaussians_hair.optimizer.step()
                gaussians_hair.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians_hair.capture(), iteration), scenehair.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    wandb.init(project="physhead-public", name=os.path.basename(args.model_path), dir="/tmp", config=vars(args))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, gaussians, gaussians_hair, dataset, pipe, background):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'color' in losses:
            tb_writer.add_scalar('train_loss_patches/color_loss', losses['color'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    wandb_scalars = {
        'train_loss_patches/l1_loss': losses['l1'].item(),
        'train_loss_patches/ssim_loss': losses['ssim'].item(),
        'train_loss_patches/total_loss': losses['total'].item(),
        'iter_time': elapsed,
    }
    if 'color' in losses:
        wandb_scalars['train_loss_patches/color_loss'] = losses['color'].item()
    wandb.log(wandb_scalars, step=iteration)

    # Report test and samples of training set — evaluated on the composited head+hair render,
    # since the head is frozen and evaluating it alone (as the old fork's training_report did)
    # would produce a metric that never changes.
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    gaussians.select_mesh_by_timestep(int(dataset.fix_tstep))
                    image = torch.clamp(merge_render(viewpoint, gaussians, gaussians_hair, pipe, background)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if (idx % (len(config['cameras']) // num_vis_img) == 0):
                        error_image = error_map(image, gt_image)
                        if tb_writer:
                            tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        wandb_images = {
                            config['name'] + "_{}/render".format(vis_ct): wandb.Image(image),
                            config['name'] + "_{}/error".format(vis_ct): wandb.Image(error_image),
                        }
                        if iteration == testing_iterations[0]:
                            wandb_images[config['name'] + "_{}/ground_truth".format(vis_ct)] = wandb.Image(gt_image)
                        wandb.log(wandb_images, step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                wandb.log({
                    config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                    config['name'] + '/loss_viewpoint - psnr': psnr_test,
                    config['name'] + '/loss_viewpoint - ssim': ssim_test,
                    config['name'] + '/loss_viewpoint - lpips': lpips_test,
                }, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", gaussians_hair.get_opacity, iteration)
            tb_writer.add_scalar('total_points', gaussians_hair.get_xyz.shape[0], iteration)
        wandb.log({
            "scene/opacity_histogram": wandb.Histogram(gaussians_hair.get_opacity.detach().cpu().numpy()),
            "total_points": gaussians_hair.get_xyz.shape[0],
        }, step=iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=5_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--head_iteration", type=int, default=600_000, help="Iteration of the pre-trained head checkpoint to load (used to resolve cfg_args; the actual weights come from --head_ply).")
    parser.add_argument("--cfg", type=str, default=None, help="Path to a YAML file with per-actor training settings; overrides matching CLI defaults.")
    args = parser.parse_args(sys.argv[1:])
    if args.cfg is not None:
        cfg = OmegaConf.load(args.cfg)
        for key, value in cfg.items():
            setattr(args, key, value)
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.head_iteration)

    # All done
    print("\nTraining complete.")

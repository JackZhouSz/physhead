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

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from gaussian_renderer import render, merge_render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer

mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh, gaussians_hair, args_cfg):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"

    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    hair_path = iter_path / "hair"

    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(hair_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []

    cfg = OmegaConf.load(args_cfg)
    frenet_folder = cfg.sim.frenet

    background = torch.tensor(cfg.bg, dtype=torch.float32, device="cuda")

    print("frenet folder", frenet_folder)

    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):

        if idx >= cfg.sim.n_frenet:
            continue

        view.image_height = 1024
        view.image_width = 1024

        # per-frame simulated hair strand geometry — overwrites the trained hair model's
        # raw tensors every frame; only SH color (learned during training) is kept fixed.
        gaussians_hair._xyz = torch.from_numpy(np.load(f'{frenet_folder}/frame_{idx + 1}_mean_frenet.npy').reshape(-1, 3)).cuda().float()
        gaussians_hair._rotation = torch.from_numpy(np.load(f'{frenet_folder}/frame_{idx + 1}_rot_frenet.npy')).cuda().float().reshape(-1, 4)
        gaussians_hair._rotation = gaussians_hair.rotation_activation(gaussians_hair._rotation)

        scales = np.load(f'{frenet_folder}/frame_{idx + 1}_scale_frenet.npy')
        new_arr = np.ones((cfg.hair.n_strand, cfg.hair.n_segment, 3)) * cfg.hair.scale
        new_arr[:, :, 0] = scales
        gaussians_hair._scaling = gaussians.scaling_inverse_activation(torch.from_numpy(new_arr.reshape(-1, 3)).cuda().float())

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)

        rendering_hair_image = render(view, gaussians_hair, pipeline, background)["render"]
        render_pkg = merge_render(view, gaussians, gaussians_hair, pipeline, background)
        image = render_pkg["render"]

        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = image
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        path2data[Path(hair_path) / f'{idx:05d}.png'] = rendering_hair_image
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader) - 1 or idx >= cfg.sim.n_frenet - 1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []

    try:
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -vf 'scale=iw-mod(iw\\,2):ih-mod(ih\\,2)' -pix_fmt yuv420p {iter_path}/renders.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -vf 'scale=iw-mod(iw\\,2):ih-mod(ih\\,2)' -pix_fmt yuv420p {iter_path}/gt.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{hair_path}/*.png' -vf 'scale=iw-mod(iw\\,2):ih-mod(ih\\,2)' -pix_fmt yuv420p {iter_path}/hair.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -vf 'scale=iw-mod(iw\\,2):ih-mod(ih\\,2)' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool, args_cfg: str):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset)
        else:
            gaussians = GaussianModel(dataset.sh_degree)

        gaussians_hair = GaussianModel(dataset.sh_degree)
        scenehair = Scene(dataset, gaussians_hair, load_iteration=iteration, shuffle=False)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dataset.target_path != "":
            name = os.path.basename(os.path.normpath(dataset.target_path))
            # when loading from a target path, test cameras are merged into the train cameras
            render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, gaussians_hair, args_cfg)
        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, gaussians_hair, args_cfg)

            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh, gaussians_hair, args_cfg)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh, gaussians_hair, args_cfg)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--cfg", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    cfg_path = getattr(args, "cfg", None)
    if cfg_path is None:
        raise ValueError("--cfg is required (path to a YAML with sim.frenet/sim.n_frenet/hair.*/bg)")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh, cfg_path)

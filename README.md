# physhead-public

This repo builds on [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) (see [README_GA.md](README_GA.md) for the base setup).

## Dataset
The tracked dataset has preprocessed sequences from [Ava-256](https://github.com/facebookresearch/ava-256). It includes multi-view captures (top row) paired with their Poisson-edited bald counterpart (bottom row), hair simulation, and tracked FLAME parameters. It's around 49GB; please contact us for more actors.


Download data-sample.zip from: https://keeper.mpdl.mpg.de/f/7150ea288f8847678486/ (or [direct download link](https://keeper.mpdl.mpg.de/seafhttp/f/7150ea288f8847678486/?op=view))

![Facial animation dataset sample](assets/jaw004_expression_grid.jpg)

## Bald-head training

`train_bald.py` is a variant of `train.py` that trains against a randomized-per-frame background instead of a fixed one, useful for bald-head reconstruction.


```shell
python train_bald.py --cfg condor_bald/RHL466_bald.yaml
```

### Cluster jobs

`condor_bald/` contains example Condor job scripts (`.sh` + `.sub` pairs) for running bald-head training on the cluster:

```shell
cd condor_bald
condor_submit_bid <bid> RHL466_bald.sub
```

### Logging

Training logs to both TensorBoard (in `<model_path>/`) and [Weights & Biases](https://wandb.ai) (project `physhead-public`) — scalars every iteration, plus a random train prediction/GT image pair and a full val/test evaluation (metrics + sample images) every `--interval` iterations.

## Hair-strand training and rendering

`train_hair_reg.py` trains only the SH color of a second, separate set of "hair" Gaussians
(initialized from an external hair-strand Frenet-frame fit) on top of a frozen, pre-trained
FLAME head, composited via `merge_render()`.

```shell
python train_hair_reg.py --cfg condor_hair/RHL466_hair_p50.yaml
```

See [`condor_hair/RHL466_hair_p50.yaml`](condor_hair/RHL466_hair_p50.yaml) for an example.
Beyond the usual `train_bald.py`-style fields, hair training additionally needs:
`head_ply` (path to the frozen head checkpoint's `point_cloud.ply`), `scale_gaus`/`rot_gaus`
(the Frenet-frame scale/rotation `.npy` fit for this actor), and `fix_tstep` (the single FLAME
timestep the head is frozen at throughout training).

### Cluster jobs

```shell
cd condor_hair
condor_submit_bid <bid> RHL466_hair_p50.sub
```

### Rendering


- **`render_together.py`** — static head+hair composite (frozen head, hair at whatever pose
  it was trained at):
  ```shell
  python render_together.py -m experiments/<actor>_hair_p50 --skip_train --skip_val --select_camera_id 0
  ```
- **`render_hair.py`** — hair-only debug view (forces the plain hair `GaussianModel`,
  ignoring the head):
  ```shell
  python render_hair.py -m experiments/<actor>_hair_p50 --skip_train --skip_val --select_camera_id 0
  ```
- **`render_hair_animation.py`** — animated hair, driven by a physically-simulated per-frame
  Frenet-frame motion sequence (`--cfg`'s `sim.frenet`/`sim.n_frenet`) while the head's
  pose/expression is driven by a separate pose-transfer dataset (`-t`/`--target_path`):
  ```shell
  ./condor_hair/render_animated.sh
  ```
  or directly:
  ```shell
  python render_hair_animation.py -m experiments/20230313--1653--RHL466_hair_p50 \
      -t <target pose-transfer dataset dir> \
      --select_camera_id 8 --iteration 10000 --cfg cfgs/RHL466_p50.yml
  ```

## Dataset

`data-sample/` includes the source data needed to run all three pipeline stages above
for one example actor (`20230313--1653--RHL466`).

```
data-sample/
└── RHL466/                                   #  20230313--1653--RHL466
    ├── facial_animation/                     # bald head training data (train_bald.py)
    │   ├── UNION10_v6_wo_20230313--1653--RHL466/   
    │   └── <8 expression sequences>/          # e.g. jaw004, lip001, cheek002, ... — each with images/ + flame_param/
    ├── hairstyle/                             # everything about one static hairstyle
    │   ├── Ava-256-2_EXP_cheek002_2/          # multi-view capture used to train hair COLOR (train_hair_reg.py --source_path)
    │   ├── geometry/                          # static Frenet-frame strand fit (scale_gaus / rot_gaus / knn_indices.npy)
    │   └── sim/                               # physically-simulated hair motion of that same static hairstyle
    │       └── rightleft/                     # one motion sequence (79 frames) — sim/ can hold more than one
    └── pose_transfer/                         # captured motion that drives the HEAD's pose during animated rendering
        └── rightleft/                         # (not the same as hairstyle/sim/rightleft/ — that's the hair's motion, this is the head's)
```


## Citation

```bibtex
@inproceedings{kabadayi2026physhead,
  title = {{PhysHead}: Simulation-Ready Gaussian Head Avatars},
  author = {Kabadayi, Berna and Sklyarova, Vanessa and Zielonka, Wojciech and Thies, Justus and Pons-Moll, Gerard},
  booktitle = {Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2026},
  month = {June}
}


```
```bibtex

@inproceedings{qian2024gaussianavatars,
  title={Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians},
  author={Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20299--20309},
  year={2024}
}


```


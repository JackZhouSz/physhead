# Preprocessing

`compute_frenet_fast.py` computes rotation-minimizing ("Frenet") frames for dense hair strands, following [Computation-of-rotation-minimizing-frames](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Computation-of-rotation-minimizing-frames.pdf). 


Given strands as a `.npy` file or a folder of them (`--n_points` points each), it saves the segment midpoints, scales and rotation quaternions as `*_mean_frenet.npy`, `*_scale_frenet.npy` and `*_rot_frenet.npy`, used for hair training and animated hair rendering in [Physhead](https://phys-head.github.io/).

```shell
python compute_frenet_fast.py --input <strands.npy or folder of .npy files> --n_points <points per strand>
```

#!/usr/bin/env python3
import numpy as np
import argparse
import os


EPS = 1e-8
# reference paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Computation-of-rotation-minimizing-frames.pdf

# --------------------------- Utility Functions --------------------------- #

def normalize(v, axis=-1, eps=EPS):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return v / norm

def quat_normalize(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, EPS)
    return q / n

def quat_mul(q1, q0):
    # Hamilton product: q = q1 * q0
    w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w = w1*w0 - x1*x0 - y1*y0 - z1*z0
    x = w1*x0 + x1*w0 + y1*z0 - z1*y0
    y = w1*y0 - x1*z0 + y1*w0 + z1*x0
    z = w1*z0 + x1*y0 - y1*x0 + z1*w0
    return np.stack((w, x, y, z), axis=-1)

def quat_from_two_vectors(a, b):
    """Return quaternion rotating vector a→b (both (...,3))."""
    a = normalize(a)
    b = normalize(b)
    dotab = np.sum(a * b, axis=-1, keepdims=True)
    w = 1.0 + dotab
    cross = np.cross(a, b)
    q = np.concatenate([w, cross], axis=-1)

    # Handle near-opposite
    near_opposite = (w[..., 0] < 1e-6)
    if np.any(near_opposite):
        a_flat = a.reshape(-1, 3)
        q_flat = q.reshape(-1, 4)
        mask = near_opposite.reshape(-1)
        for i in np.where(mask)[0]:
            av = a_flat[i]
            if abs(av[0]) < 0.9:
                orth = np.array([1.0, 0.0, 0.0])
            else:
                orth = np.array([0.0, 1.0, 0.0])
            axis = np.cross(av, orth)
            axis /= np.linalg.norm(axis) + EPS
            q_flat[i, 0] = 0.0
            q_flat[i, 1:] = axis
        q = q_flat.reshape(q.shape)

    return quat_normalize(q)

def ensure_quat_continuity(quats):
    """Flip quaternion signs to avoid 180° flips."""
    q = quats.copy()
    if q.ndim == 2:
        for i in range(1, q.shape[0]):
            if np.dot(q[i-1], q[i]) < 0:
                q[i] = -q[i]
    else:
        for b in range(q.shape[0]):
            for i in range(1, q.shape[1]):
                if np.dot(q[b, i-1], q[b, i]) < 0:
                    q[b, i] = -q[b, i]
    return q

# --------------------------- RMF Computation --------------------------- #

def rotation_minimizing_frames(points):
    """
    Pure NumPy vectorized RMF computation (no Python loops).
    points: (B, N, 3)
    Returns tangents, normals, binormals, quaternions
    """
    EPS = 1e-8
    if points.ndim == 2:
        points = points[np.newaxis, ...]

    B, N, _ = points.shape
    M = N - 1

    # Tangents
    tang = points[:, 1:, :] - points[:, :-1, :]
    tang = tang / (np.linalg.norm(tang, axis=-1, keepdims=True) + EPS)

    # Reference vectors for initial frame (avoid parallel)
    ref = np.where(np.abs(tang[:, 0:1, 0:1]) < 0.9,
                   np.array([1., 0., 0.])[None, None, :],
                   np.array([0., 1., 0.])[None, None, :])

    # Initial normals and binormals
    n0 = np.cross(tang[:, 0:1, :], ref)
    n0 /= (np.linalg.norm(n0, axis=-1, keepdims=True) + EPS)
    b0 = np.cross(tang[:, 0:1, :], n0)

    # Prepare arrays
    normals = np.zeros_like(tang)
    binormals = np.zeros_like(tang)
    normals[:, 0:1, :] = n0
    binormals[:, 0:1, :] = b0

    # Compute rotation axes and angles for all segments
    v = tang[:, :-1, :]
    w = tang[:, 1:, :]
    axis = np.cross(v, w)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.divide(axis, axis_norm + EPS)
    cos_theta = np.sum(v * w, axis=-1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Rodrigues’ rotation of normals
    n_prev = normals[:, :-1, :]
    c, s = np.cos(theta), np.sin(theta)
    dot_an = np.sum(axis * n_prev, axis=-1, keepdims=True)
    n_rot = n_prev * c + np.cross(axis, n_prev) * s + axis * dot_an * (1 - c)
    n_rot /= (np.linalg.norm(n_rot, axis=-1, keepdims=True) + EPS)

    # Fill normals and binormals (vectorized cumulative update)
    normals[:, 1:, :] = np.concatenate([n_rot[:, 0:1, :], n_rot], axis=1)[:, :-1, :]
    binormals = np.cross(tang, normals)
    binormals /= (np.linalg.norm(binormals, axis=-1, keepdims=True) + EPS)

    # Build quaternions from tangent/normal/binormal
    t, n, b = tang, normals, binormals
    R = np.stack([t, n, b], axis=-1)  # (B, M, 3, 3)

    # Quaternion from rotation matrix
    qw = np.sqrt(1.0 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] + EPS) / 2.0
    qx = (R[..., 2, 1] - R[..., 1, 2]) / (4 * qw + EPS)
    qy = (R[..., 0, 2] - R[..., 2, 0]) / (4 * qw + EPS)
    qz = (R[..., 1, 0] - R[..., 0, 1]) / (4 * qw + EPS)
    quats = np.stack([qw, qx, qy, qz], axis=-1)
    quats /= (np.linalg.norm(quats, axis=-1, keepdims=True) + EPS)

    return tang, normals, binormals, quats

# --------------------------- Hair Strand Processing --------------------------- #

def calculate_pts_scal(hair_strands):
    midpoints = (hair_strands[:, :-1, :] + hair_strands[:, 1:, :]) / 2
    scales = np.linalg.norm(hair_strands[:, 1:, :] - hair_strands[:, :-1, :], axis=2) / 2.0
    return midpoints, scales


def calculate_pts_scal_dominant_scale(hair_strands):
    """
    Compute midpoints and per-segment 'dominant-axis' scales.

    Instead of the Euclidean distance, the scale is taken as the
    largest absolute displacement along X, Y, or Z for each segment.
    """
    # midpoint between each consecutive pair of strand points
    midpoints = (hair_strands[:, :-1, :] + hair_strands[:, 1:, :]) / 2

    # absolute per-axis displacements
    diff = np.abs(hair_strands[:, 1:, :] - hair_strands[:, :-1, :])

    # dominant (maximum) displacement across x/y/z axes
    scales = np.max(diff, axis=2)

    return midpoints, scales

def calculate_rot_quat(hair_strands):
    """Compute stable RMF quaternions for each hair strand."""
    if hair_strands.ndim == 2:
        hair_strands = hair_strands[np.newaxis, ...]
    batch = hair_strands.shape[0]
    all_quats = []
    for b in range(batch):
        _, _, _, quats = rotation_minimizing_frames(hair_strands[b])
        all_quats.append(quats)
    return np.stack(all_quats, axis=0)

# --------------------------- I/O + Main --------------------------- #

def calculate_frenet_frame_t(inp_strands, args):
    hair_strand_points = np.load(inp_strands).reshape(-1, args.n_points, 3)
    print('Strands shape:', hair_strand_points.shape)

    midpoints, scales = calculate_pts_scal(hair_strand_points)
    quats = calculate_rot_quat(hair_strand_points)

    print('Rotation Quaternions:', quats.shape)
    print('Scales:', scales.shape)
    print('Midpoints:', midpoints.shape)

    mean_frenet = inp_strands.replace('.npy', '_mean_frenet.npy')
    rot_frenet = inp_strands.replace('.npy', '_rot_frenet.npy')
    scale_frenet = inp_strands.replace('.npy', '_scale_frenet.npy')

    np.save(mean_frenet, midpoints)
    np.save(rot_frenet, quats)
    np.save(scale_frenet, scales)

def main(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    if args.input.endswith('.npy'):
        print('Computing RMF frames for a single file...')
        calculate_frenet_frame_t(args.input, args)
    else:
        frames = sorted([f for f in os.listdir(args.input) if f.endswith('.npy')])
        print(f'Found {len(frames)} frames.')
        for i, f in enumerate(frames):
            print(f'Frame {i+1}/{len(frames)}: {f}')
            calculate_frenet_frame_t(os.path.join(args.input, f), args)
    return 0

# --------------------------- Entry Point --------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to a .npy file or directory containing dense hair strands for simulation')
    parser.add_argument('--n_points', default=16, type=int)
    parser.add_argument('--rot_format', choices=['quat', 'mat'],
                        default='quat', help='Rotation output format (quat recommended).')
    args = parser.parse_args()
    main(args)

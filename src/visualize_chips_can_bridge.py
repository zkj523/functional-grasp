#!/usr/bin/env python3
"""Create reviewable PNGs for chips-can registration and affordance selection."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers projection="3d"
import numpy as np
import yaml


def equal_axes(ax, points):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.55 * float((hi - lo).max())
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    # Matplotlib 3.3+ supports true 3-D box aspect. Ubuntu/ROS hosts may ship
    # an older release; equal numerical axis limits still preserve the review.
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.registration_dir
    reg = yaml.safe_load((root / "registration.yaml").read_text(encoding="utf-8"))
    observed_camera = np.load(root / "observed_partial_512x6_camera.npy")
    model_base = np.load(root / "ycb_complete_512x6_base.npy")
    base_t_camera = np.asarray(reg["base_T_camera"]["matrix_4x4"], dtype=np.float64)
    observed_base = (base_t_camera[:3, :3] @ observed_camera[:, :3].T).T + base_t_camera[:3, 3]

    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    for i, (elev, azim, title) in enumerate(((24, -55, "oblique"), (88, -90, "top")), 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(observed_base[:, 0], observed_base[:, 1], observed_base[:, 2],
                   s=8, c="#20c878", label="observed RGB-D")
        ax.scatter(model_base[:, 0], model_base[:, 1], model_base[:, 2],
                   s=8, c="#ff8c2a", alpha=0.72, label="registered YCB")
        center = np.asarray(reg["object_center_base_m"])
        axis = np.asarray(reg["object_axis_base"])
        endpoints = np.stack((center - 0.125 * axis, center + 0.125 * axis))
        ax.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], c="black", lw=2)
        equal_axes(ax, np.vstack((observed_base, model_base[:, :3])))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        ax.set_zlabel("base z (m)")
    fig.axes[0].legend(loc="upper left")
    fig.suptitle("Chips-can registration review")
    fig.savefig(root / "registration_review.png", dpi=180)
    plt.close(fig)

    affordance_dir = root / "affordance"
    scores = np.load(affordance_dir / "affordance_scores.npz")
    selected = json.loads((affordance_dir / "selected_affordance.json").read_text(encoding="utf-8"))
    xyz = model_base[:, :3]
    fusion = scores["fusion"]
    middle = scores["middle_mask"].astype(bool)
    index = int(selected["selected_index"])

    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    for i, (elev, azim, title) in enumerate(((24, -55, "oblique"), (88, -90, "top")), 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(xyz[~middle, 0], xyz[~middle, 1], xyz[~middle, 2],
                   s=10, c="#a8a8a8", alpha=0.12)
        scatter = ax.scatter(xyz[middle, 0], xyz[middle, 1], xyz[middle, 2], s=18,
                             c=fusion[middle], cmap="viridis", vmin=0, vmax=1, alpha=1.0)
        ax.scatter(*xyz[index], s=180, marker="*", c="white", edgecolors="black", linewidths=1.2)
        equal_axes(ax, xyz)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        ax.set_zlabel("base z (m)")
    fig.colorbar(scatter, ax=fig.axes, shrink=0.78, label="fused affordance")
    fig.suptitle("Middle-body affordance review (faded points are outside middle constraint)")
    fig.savefig(affordance_dir / "affordance_review.png", dpi=180)
    plt.close(fig)

    candidate_path = root / "ag95_plan_only_candidates_base.json"
    if candidate_path.exists():
        candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidates = candidate_data["candidates"][:5]
        fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=12, c="#ff9a33", alpha=0.65)
        selected_xyz = np.asarray(candidate_data["selected_affordance_base_m"])
        ax.scatter(*selected_xyz, s=220, marker="*", c="red", edgecolors="black")
        all_points = [xyz]
        for rank, candidate in enumerate(candidates):
            p = np.array([candidate["position"][k] for k in ("x", "y", "z")])
            q = np.array([candidate["orientation"][k] for k in ("x", "y", "z", "w")])
            rotation = quat_matrix(q)
            color = "#1769ff" if rank == 0 else "#6aa5ff"
            ax.scatter(*p, s=70 if rank == 0 else 35, c=color)
            ax.quiver(*p, *(rotation[:, 0]*0.08), color="#d62728", linewidth=2)
            ax.quiver(*p, *(rotation[:, 1]*0.08), color="#1769ff", linewidth=2)
            ax.plot([p[0], selected_xyz[0]], [p[1], selected_xyz[1]],
                    [p[2], selected_xyz[2]], c="gray", ls="--", alpha=0.6)
            all_points.append(p[None])
        equal_axes(ax, np.vstack(all_points))
        ax.view_init(elev=25, azim=-55)
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        ax.set_zlabel("base z (m)")
        ax.set_title("Top 5 AG95 PLAN-ONLY pregrasps\nred axis=approach, blue axis=jaw opening")
        fig.savefig(root / "ag95_plan_only_review.png", dpi=180)
        plt.close(fig)

    print(root / "registration_review.png")
    print(affordance_dir / "affordance_review.png")
    if candidate_path.exists():
        print(root / "ag95_plan_only_review.png")


if __name__ == "__main__":
    main()

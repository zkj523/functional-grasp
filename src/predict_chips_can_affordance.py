#!/usr/bin/env python3
"""Run the paper's Score/PointNet fusion on a registered YCB chips can."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ral_affordance_pointnet", REPO_ROOT / "tasks" / "affordance_pointnet.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
PointNetAffordance = _module.PointNetAffordance


def normalize_scores(x):
    x=np.asarray(x,dtype=np.float64)
    span=float(x.max()-x.min())
    return np.zeros_like(x) if span<1e-8 else (x-x.min())/span


def quat_matrix(q):
    x,y,z,w=np.asarray(q,dtype=np.float64)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def write_score_ply(path,xyz,scores,selected):
    scores=normalize_scores(scores)
    colors=np.column_stack((255*scores,80+120*(1-scores),255*(1-scores))).astype(np.uint8)
    colors[selected]=[255,255,255]
    with Path(path).open("w",encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p,c in zip(xyz,colors):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {c[0]} {c[1]} {c[2]}\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--registration",type=Path,required=True)
    ap.add_argument("--points",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--alpha",type=float,default=0.75)
    ap.add_argument("--middle-fraction",type=float,default=0.60)
    ap.add_argument("--topk-ratio",type=float,default=0.20)
    ap.add_argument("--selection-mode",choices=("sample","argmax"),default="sample",
                    help="Keep sample for paper-compatible evaluation; use argmax for deterministic real execution.")
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args()

    reg=yaml.safe_load(args.registration.read_text(encoding="utf-8"))
    points=np.load(args.points).astype(np.float32)
    if points.shape!=(512,6): raise ValueError(f"Expected 512x6, got {points.shape}")
    t=np.asarray(reg["base_T_object"]["matrix_4x4"],dtype=np.float64)
    r=t[:3,:3]
    xyz_base=(r@points[:,:3].T).T+t[:3,3]
    normals_base=(r@points[:,3:].T).T

    # This mirrors tasks/grasp.py::compute_score_based_affordance_scores.
    height=(xyz_base[:,2]>t[2,3]).astype(np.float64)
    cosine=np.clip(normals_base[:,2],-1,1)
    angle=np.arccos(cosine)
    score_raw=height+np.clip(1-angle/(np.pi/2),0,1)

    model=PointNetAffordance().to(args.device)
    ckpt=torch.load(args.checkpoint,map_location=args.device)
    model.load_state_dict(ckpt.get("model_state_dict",ckpt)); model.eval()
    with torch.no_grad():
        pointnet=model(torch.from_numpy(points).unsqueeze(0).to(args.device))[0].cpu().numpy()
    score=normalize_scores(score_raw); pointnet_norm=normalize_scores(pointnet)
    fusion=args.alpha*score+(1-args.alpha)*pointnet_norm

    # Canonical local Z is the can long axis. Keep only the requested body middle.
    axis_coord=points[:,2].astype(np.float64)
    axis_center=0.5*(axis_coord.min()+axis_coord.max())
    half_window=0.5*args.middle_fraction*(axis_coord.max()-axis_coord.min())
    middle=np.abs(axis_coord-axis_center)<=half_window
    valid=np.flatnonzero(middle)
    k=max(1,min(len(valid),int(round(len(points)*args.topk_ratio))))
    top=valid[np.argsort(fusion[valid])[-k:]]
    if args.selection_mode == "argmax":
        selected=int(top[np.argmax(fusion[top])])
    else:
        probs=np.maximum(fusion[top],1e-6); probs/=probs.sum()
        selected=int(np.random.default_rng(args.seed).choice(top,p=probs))

    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez(args.output_dir/"affordance_scores.npz",score=score,pointnet=pointnet_norm,
             fusion=fusion,middle_mask=middle,topk_indices=top,selected_index=selected)
    write_score_ply(args.output_dir/"fusion_affordance_base.ply",xyz_base,fusion,selected)
    result={
        "schema":"chips_can_middle_affordance_v1","alpha_score":args.alpha,
        "alpha_pointnet":1-args.alpha,"middle_fraction":args.middle_fraction,
        "topk_ratio":args.topk_ratio,"selection_mode":args.selection_mode,"selected_index":selected,
        "selected_point_object_m":points[selected,:3].astype(float).tolist(),
        "selected_normal_object":points[selected,3:].astype(float).tolist(),
        "selected_point_base_m":xyz_base[selected].astype(float).tolist(),
        "selected_normal_base":normals_base[selected].astype(float).tolist(),
        "selected_scores":{"score":float(score[selected]),"pointnet":float(pointnet_norm[selected]),
                           "fusion":float(fusion[selected])},
        "middle_axis_range_object_m":[float(axis_center-half_window),float(axis_center+half_window)],
        "checkpoint":str(args.checkpoint),"registration":str(args.registration),
    }
    (args.output_dir/"selected_affordance.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()

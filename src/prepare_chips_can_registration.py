#!/usr/bin/env python3
"""Register a side-lying YCB chips can to a synchronized real RGB-D capture.

The chips can is rotationally symmetric, so this estimates the observable pose:
the can center, its long axis, and the table-up radial direction. Rotation around
the can axis is intentionally fixed by the table normal, not by package texture.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree


def quat_xyzw_to_matrix(q):
    x, y, z, w = [float(v) for v in q]
    n = np.linalg.norm([x, y, z, w])
    x, y, z, w = np.asarray([x, y, z, w]) / n
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ], dtype=np.float64,
    )


def matrix_to_quat_xyzw(r):
    trace = float(np.trace(r))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s,
                      (r[1,0]-r[0,1])/s, 0.25*s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1+r[0,0]-r[1,1]-r[2,2]) * 2
            q = np.array([0.25*s, (r[0,1]+r[1,0])/s,
                          (r[0,2]+r[2,0])/s, (r[2,1]-r[1,2])/s])
        elif i == 1:
            s = np.sqrt(1+r[1,1]-r[0,0]-r[2,2]) * 2
            q = np.array([(r[0,1]+r[1,0])/s, 0.25*s,
                          (r[1,2]+r[2,1])/s, (r[0,2]-r[2,0])/s])
        else:
            s = np.sqrt(1+r[2,2]-r[0,0]-r[1,1]) * 2
            q = np.array([(r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s,
                          0.25*s, (r[1,0]-r[0,1])/s])
    return q / np.linalg.norm(q)


def pose_matrix(pose):
    p, q = pose["position_m"], pose["orientation_xyzw"]
    t = np.eye(4)
    t[:3, :3] = quat_xyzw_to_matrix([q["x"], q["y"], q["z"], q["w"]])
    t[:3, 3] = [p["x"], p["y"], p["z"]]
    return t


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / n


def largest_component(mask, min_area=500):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                  if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not candidates:
        raise RuntimeError("No chips-can color component found")
    return labels == max(candidates)[1]


def chips_can_mask(rgb_bgr, depth_m):
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    green = ((hsv[...,0] >= 34) & (hsv[...,0] <= 100) &
             (hsv[...,1] >= 35) & (hsv[...,2] >= 30))
    red = (((hsv[...,0] <= 12) | (hsv[...,0] >= 165)) &
           (hsv[...,1] >= 45) & (hsv[...,2] >= 40))
    yellow = ((hsv[...,0] >= 12) & (hsv[...,0] <= 34) &
              (hsv[...,1] >= 45) & (hsv[...,2] >= 45))
    mask = (green | red | yellow) & (depth_m > 0.15) & (depth_m < 1.2)
    u8 = mask.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8), iterations=2)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = largest_component(u8 > 0)
    z = depth_m[mask]
    med = float(np.median(z))
    mask &= (depth_m > med - 0.07) & (depth_m < med + 0.07)
    return largest_component(mask), med


def unproject(depth_m, mask, k, stride=1):
    rows, cols = np.where(mask)
    if stride > 1:
        rows, cols = rows[::stride], cols[::stride]
    z = depth_m[rows, cols].astype(np.float64)
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    xyz = np.column_stack(((cols-cx)*z/fx, (rows-cy)*z/fy, z))
    return xyz, rows, cols


def transform_points(t, xyz):
    return (t[:3,:3] @ xyz.T).T + t[:3,3]


def ransac_plane(points, seed=7, threshold=0.0035, iterations=800):
    rng = np.random.default_rng(seed)
    if len(points) > 60000:
        points = points[rng.choice(len(points), 60000, replace=False)]
    best = None
    for _ in range(iterations):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1]-p[0], p[2]-p[0])
        if np.linalg.norm(n) < 1e-8:
            continue
        n = normalize(n)
        if abs(n[2]) < 0.85:
            continue
        d = -float(n @ p[0])
        inliers = np.abs(points @ n + d) < threshold
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, n, d, inliers)
    if best is None:
        raise RuntimeError("Could not fit tabletop plane")
    _, _, _, inliers = best
    c = points[inliers].mean(0)
    _, _, vh = np.linalg.svd(points[inliers]-c, full_matrices=False)
    n = vh[-1]
    if n[2] < 0:
        n = -n
    d = -float(n @ c)
    return n, d, int(inliers.sum()), float(np.sqrt(np.mean((points[inliers]@n+d)**2)))


def estimate_normals(points, camera_origin, k=24):
    tree = cKDTree(points)
    _, neighbors = tree.query(points, k=min(k, len(points)))
    normals = np.empty_like(points)
    for i, idx in enumerate(neighbors):
        local = points[idx]
        cov = np.cov((local-local.mean(0)).T)
        _, vec = np.linalg.eigh(cov)
        normal = vec[:,0]
        if normal @ (camera_origin-points[i]) < 0:
            normal = -normal
        normals[i] = normalize(normal)
    return normals


def farthest_sample(points, count=512, seed=7):
    rng = np.random.default_rng(seed)
    if len(points) <= count:
        return rng.choice(len(points), count, replace=len(points)<count)
    out = np.empty(count, np.int64)
    out[0] = rng.integers(len(points))
    distance = np.full(len(points), np.inf)
    for i in range(1, count):
        delta = points-points[out[i-1]]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        out[i] = int(np.argmax(distance))
    return out


def write_ply(path, groups):
    # groups: [(xyz, rgb)]
    total = sum(len(x) for x, _ in groups)
    with Path(path).open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in groups:
            color = np.broadcast_to(np.asarray(rgb, np.uint8), (len(xyz),3))
            for p, c in zip(xyz, color):
                f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {c[0]} {c[1]} {c[2]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--diameter", type=float, default=0.075)
    ap.add_argument("--length", type=float, default=0.250)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    base_dir = args.metadata.parent
    rgb = cv2.imread(str(base_dir/meta["saved_files"]["rgb_png"]), cv2.IMREAD_COLOR)
    depth_raw = np.load(base_dir/meta["saved_files"]["depth_npy"])
    if rgb is None or depth_raw.shape != rgb.shape[:2]:
        raise RuntimeError("RGB/depth capture is missing or has inconsistent shape")
    depth_m = depth_raw.astype(np.float64)*0.001
    k = meta["camera_info"]["color"]["K"]
    base_t_camera = pose_matrix(meta["camera_pose"])

    mask, median_depth = chips_can_mask(rgb, depth_m)
    object_camera, rows, cols = unproject(depth_m, mask, k)
    object_base = transform_points(base_t_camera, object_camera)

    ys, xs = np.where(mask)
    x0, x1 = max(0,xs.min()-180), min(mask.shape[1],xs.max()+181)
    y0, y1 = max(0,ys.min()-160), min(mask.shape[0],ys.max()+161)
    roi = np.zeros_like(mask)
    roi[y0:y1,x0:x1] = True
    exclusion = cv2.dilate(mask.astype(np.uint8), np.ones((51,51),np.uint8), iterations=1)>0
    table_mask = roi & ~exclusion & (depth_m>0.2) & (depth_m<1.2)
    table_camera, _, _ = unproject(depth_m, table_mask, k, stride=2)
    table_base = transform_points(base_t_camera, table_camera)
    plane_n, plane_d, plane_count, plane_rmse = ransac_plane(table_base, args.seed)

    height = object_base@plane_n+plane_d
    keep = (height>0.004) & (height<args.diameter*1.15)
    object_base = object_base[keep]
    object_camera = object_camera[keep]
    rows, cols = rows[keep], cols[keep]
    if len(object_base)<512:
        raise RuntimeError(f"Only {len(object_base)} object points survived plane filtering")

    rough_center = object_base.mean(0)
    _, _, vh = np.linalg.svd(object_base-rough_center, full_matrices=False)
    can_axis = vh[0]-plane_n*np.dot(vh[0],plane_n)
    can_axis = normalize(can_axis)
    if can_axis[0]<0:
        can_axis=-can_axis
    s = object_base@can_axis
    lo, hi = np.percentile(s,[2.5,97.5])
    axis_mid = 0.5*(lo+hi)
    c = object_base.mean(0)+can_axis*(axis_mid-object_base.mean(0)@can_axis)
    c_on_plane = c-plane_n*(c@plane_n+plane_d)
    physical_center = c_on_plane+plane_n*(args.diameter*0.5)

    # Canonical local Z is the can long axis. Local X is fixed to table-up.
    z_axis = can_axis
    x_axis = normalize(plane_n-z_axis*np.dot(plane_n,z_axis))
    y_axis = normalize(np.cross(z_axis,x_axis))
    x_axis = normalize(np.cross(y_axis,z_axis))
    base_r_object = np.column_stack((x_axis,y_axis,z_axis))

    canonical = np.load(args.canonical).astype(np.float64)
    if canonical.shape!=(512,6):
        raise ValueError(f"Expected canonical 512x6, got {canonical.shape}")
    canonical_center = 0.5*(canonical[:,:3].min(0)+canonical[:,:3].max(0))
    base_t_object = np.eye(4)
    base_t_object[:3,:3] = base_r_object
    base_t_object[:3,3] = physical_center-base_r_object@canonical_center
    camera_t_object = np.linalg.inv(base_t_camera)@base_t_object

    model_base = transform_points(base_t_object,canonical[:,:3])
    model_normals_base = (base_r_object@canonical[:,3:].T).T
    distances = cKDTree(model_base).query(object_base,k=1)[0]

    normals_camera = estimate_normals(object_camera,np.zeros(3))
    idx = farthest_sample(object_camera,512,args.seed)
    observed_512 = np.column_stack((object_camera[idx],normals_camera[idx])).astype(np.float32)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.save(args.output_dir/"observed_partial_512x6_camera.npy",observed_512)
    np.save(args.output_dir/"ycb_complete_512x6_object.npy",canonical.astype(np.float32))
    np.save(args.output_dir/"ycb_complete_512x6_base.npy",
            np.column_stack((model_base,model_normals_base)).astype(np.float32))
    np.save(args.output_dir/"affordance_input_512x6_object.npy",canonical.astype(np.float32))

    overlay=rgb.copy()
    tint=overlay.copy(); tint[mask]=(0,220,0)
    overlay=cv2.addWeighted(tint,0.4,overlay,0.6,0)
    cv2.imwrite(str(args.output_dir/"segmentation_overlay.png"),overlay)
    write_ply(args.output_dir/"registration_overlay_base.ply",
              [(model_base,(30,144,255)),(object_base[::max(1,len(object_base)//4000)],(0,255,80))])

    def tf_record(t):
        q=matrix_to_quat_xyzw(t[:3,:3])
        return {"translation_m":[float(v) for v in t[:3,3]],
                "quaternion_xyzw":[float(v) for v in q],
                "matrix_4x4":t.tolist()}
    report={
        "schema":"chips_can_ycb_registration_v1",
        "source_metadata":str(args.metadata),
        "canonical_pointcloud":str(args.canonical),
        "symmetry_note":"Rotation around the cylindrical long axis is fixed by table-up and is not texture-observed.",
        "dimensions_m":{"diameter":args.diameter,"length":args.length},
        "rgb_depth_sync_delta_sec":float(meta["sync_delta_sec"]),
        "median_object_depth_m":median_depth,
        "table_plane_base_abcd":[*map(float,plane_n),float(plane_d)],
        "table_plane_inliers":plane_count,
        "table_plane_rmse_m":plane_rmse,
        "object_visible_points":int(len(object_base)),
        "observed_axis_span_m":float(hi-lo),
        "object_center_base_m":[float(v) for v in physical_center],
        "object_axis_base":[float(v) for v in can_axis],
        "base_T_camera":tf_record(base_t_camera),
        "base_T_object":tf_record(base_t_object),
        "camera_T_object":tf_record(camera_t_object),
        "observed_to_model_distance_m":{
            "mean":float(distances.mean()),"median":float(np.median(distances)),
            "p90":float(np.percentile(distances,90)),"p95":float(np.percentile(distances,95))},
    }
    (args.output_dir/"registration.yaml").write_text(yaml.safe_dump(report,sort_keys=False),encoding="utf-8")
    print(yaml.safe_dump(report,sort_keys=False))


if __name__=="__main__":
    main()

#!/usr/bin/env python
"""Convert RoboTwin 2.0 raw hdf5 to LeRobot v3 — one task×setting per dataset.

Reverse-engineered from hxma/RoboTwin-LeRobot-v3.0 (verified frame-exact):
  observation.state[t] = joint_action/vector[t]      (current 14-dim qpos)
  action[t]            = joint_action/vector[t+1]     (next-state-as-action;
                          RoboTwin uses absolute joint position control)
  → drop the final frame (no t+1 action), so T_out = T_raw - 1.

Cameras: head→cam_high, left→cam_left_wrist, right→cam_right_wrist (front
dropped, matching hxma + pi05 configs). RGB stored at the sim's native
240×320 (hxma upscaled to 480×640, which adds no information). JPEG bytes in
the hdf5 are cv2-decoded BGR→RGB.

Instruction: one per episode, deterministically sampled from the 100 `seen`
prompts (episode-index-seeded). `unseen` prompts are left for eval-time
language-generalization tests.

Stats: lerobot's create→save_episode does NOT compute quantiles. q01/q99 must
be added afterward (scripts/augment_robotwin_quantile_stats.py for per-task;
scripts/compute_robotwin_pooled_stats.py for the merged global stats).

Usage (single task×setting):
    python scripts/convert_robotwin_to_lerobot.py \
        --task beat_block_hammer --setting aloha-agilex_clean_50 \
        --raw-root /path/to/RoboTwin2.0-raw/dataset \
        --out-root /path/to/RoboTwin-LeRobot-v3.0
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]

# RoboTwin raw camera name -> LeRobot/pi05 camera key
CAM_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def episode_num(p: Path) -> int:
    return int(p.stem.replace("episode", ""))


def decode_camera(ep: h5py.File, raw_cam: str) -> np.ndarray:
    """Decode a camera's JPEG byte sequence to (T, H, W, 3) RGB uint8."""
    frames = []
    for buf in ep[f"observation/{raw_cam}/rgb"]:
        bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def load_episode(ep_path: Path):
    """Return (state[T-1,14], action[T-1,14], {cam: imgs[T-1,H,W,3]})."""
    with h5py.File(ep_path, "r") as ep:
        ja = ep["joint_action/vector"][:].astype(np.float32)  # (T, 14)
        imgs = {lr: decode_camera(ep, raw) for raw, lr in CAM_MAP.items()}
    state = ja[:-1]
    action = ja[1:]
    imgs = {k: v[:-1] for k, v in imgs.items()}  # align to state (drop last)
    return state, action, imgs


def pick_instruction(ep_path: Path, ep_idx: int) -> str:
    """Deterministically sample one `seen` instruction for this episode."""
    instr = json.loads((ep_path.parent.parent / "instructions" / f"episode{ep_idx}.json").read_text())
    seen = instr["seen"]
    rng = np.random.RandomState(ep_idx)
    return str(rng.choice(seen))


def build_features(h: int, w: int) -> dict:
    feats = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": {"motors": MOTORS}},
        "action": {"dtype": "float32", "shape": (14,), "names": {"motors": MOTORS}},
    }
    for cam in CAM_MAP.values():
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "rgb"],
        }
    return feats


def convert(task: str, setting: str, raw_root: Path, out_root: Path,
            image_writer_threads: int = 4, image_writer_processes: int = 0) -> None:
    raw_dir = raw_root / task / setting
    hdf5_files = sorted((raw_dir / "data").glob("episode*.hdf5"), key=episode_num)
    if not hdf5_files:
        raise FileNotFoundError(f"No episode*.hdf5 under {raw_dir/'data'}")

    # Peek first episode for image dims.
    _, _, imgs0 = load_episode(hdf5_files[0])
    h, w = imgs0["cam_high"].shape[1:3]

    out_path = out_root / task / setting
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset.create(
        repo_id=f"robotwin/{task}_{setting}",
        fps=30,
        features=build_features(h, w),
        root=out_path,
        robot_type="aloha",
        use_videos=True,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    for ep_path in hdf5_files:
        ep_idx = episode_num(ep_path)
        state, action, imgs = load_episode(ep_path)
        instruction = pick_instruction(ep_path, ep_idx)
        for t in range(len(state)):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "task": instruction,
            }
            for cam, arr in imgs.items():
                frame[f"observation.images.{cam}"] = arr[t]
            ds.add_frame(frame)
        ds.save_episode()

    ds.finalize()
    print(f"[done] {task}/{setting}: {len(hdf5_files)} episodes → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--setting", required=True,
                    choices=["aloha-agilex_clean_50", "aloha-agilex_randomized_500"])
    ap.add_argument("--raw-root", type=Path, required=True,
                    help="Root of the extracted raw RoboTwin 2.0 dataset (contains <task>/<setting>/data/*.hdf5).")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Output root for the converted LeRobot v3 tree (<task>/<setting>/...).")
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--image-writer-processes", type=int, default=0)
    args = ap.parse_args()
    convert(args.task, args.setting, args.raw_root, args.out_root,
            args.image_writer_threads, args.image_writer_processes)


if __name__ == "__main__":
    main()

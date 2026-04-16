"""
将 RoboTwin 原始数据转换为 Cosmos-Predict2.5 Action-Conditioned 训练格式。

V2 版本变化：
- 切分逻辑：**按 task prefix 各自切 90/10**（不是全局切）
  原脚本会让 pcp 100 个全进 train、pob 80 train + 20 val，val 任务不均衡
  本版本：pcp 90 train + 10 val、pob 90 train + 10 val
- 输出路径改成 -v2 后缀，旧数据保留

用法：
    python scripts/convert_robotwin_to_cosmos_balanced.py

输入：/mnt/gyc/RoboTwin_ckp/data/{task}/region_all_clean/
输出：/mnt/gyc_ckp/cosmos-predict2.5-robotwin-v2/
"""

import json
import numpy as np
import h5py
import transforms3d as t3d
from pathlib import Path
from tqdm import tqdm

# ─── 路径配置 ───────────────────────────────────────────────────────────────
INPUT_BASE  = Path("/mnt/gyc/RoboTwin_ckp/data")
OUTPUT_BASE = Path("/mnt/gyc_ckp/cosmos-predict2.5-robotwin-v2")

TASKS = {
    "pcp": "place_container_plate",
    "pob": "place_object_basket",
}
REGION = "region_all_clean"

VAL_RATIO = 0.1   # 每个 task 各自切 10% 作为 val
# ────────────────────────────────────────────────────────────────────────────


def quat_to_euler(endpose: np.ndarray) -> np.ndarray:
    """[x, y, z, qw, qx, qy, qz] → [x, y, z, roll, pitch, yaw]"""
    xyz  = endpose[:3]
    quat = endpose[3:]
    rotm = t3d.quaternions.quat2mat(quat)
    rpy  = t3d.euler.mat2euler(rotm, axes='sxyz')
    return np.concatenate([xyz, np.array(rpy)])


def convert_episode(hdf5_path: Path, mp4_src: Path, mp4_dst: Path,
                    json_dst: Path, episode_id: str, instruction: str):
    """读一个 episode 的 HDF5，生成 JSON + 软链接 MP4。"""
    with h5py.File(hdf5_path, "r") as f:
        left_endpose  = f["endpose/left_endpose"][:]
        right_endpose = f["endpose/right_endpose"][:]
        left_gripper  = f["endpose/left_gripper"][:]
        right_gripper = f["endpose/right_gripper"][:]

    n_frames = left_endpose.shape[0]

    state = []
    for i in range(n_frames):
        left_euler  = quat_to_euler(left_endpose[i])
        right_euler = quat_to_euler(right_endpose[i])
        row = np.concatenate([
            left_euler,
            right_euler,
            [left_gripper[i], right_gripper[i]],
        ])
        state.append(row.tolist())

    continuous_gripper_state = [
        float((left_gripper[i] + right_gripper[i]) / 2.0)
        for i in range(n_frames)
    ]

    video_rel = str(mp4_dst.relative_to(OUTPUT_BASE))

    annotation = {
        "texts":                    [instruction],
        "videos":                   [{"video_path": video_rel}],
        "state":                    state,
        "continuous_gripper_state": continuous_gripper_state,
        "episode_id":               episode_id,
    }

    json_dst.parent.mkdir(parents=True, exist_ok=True)
    with open(json_dst, "w") as f:
        json.dump(annotation, f)

    mp4_dst.parent.mkdir(parents=True, exist_ok=True)
    if mp4_dst.exists() or mp4_dst.is_symlink():
        mp4_dst.unlink()
    mp4_dst.symlink_to(mp4_src.resolve())


def collect_task_episodes(prefix: str, task_name: str):
    """收集一个 task 的所有 episode，返回排序后的列表。"""
    region_dir = INPUT_BASE / task_name / REGION
    data_dir   = region_dir / "data"
    video_dir  = region_dir / "video"
    instr_dir  = region_dir / "instructions"

    hdf5_files = sorted(data_dir.glob("episode*.hdf5"),
                        key=lambda p: int(p.stem.replace("episode", "")))

    episodes = []
    for hdf5_path in hdf5_files:
        idx = int(hdf5_path.stem.replace("episode", ""))
        mp4_src = video_dir / f"episode{idx}.mp4"
        instr_file = instr_dir / f"episode{idx}.json"

        if not mp4_src.exists():
            print(f"  [WARN] 缺少视频：{mp4_src}，跳过")
            continue

        instruction = "robot manipulation task"
        if instr_file.exists():
            with open(instr_file) as f:
                instr_data = json.load(f)
            seen = instr_data.get("seen", [])
            if seen:
                instruction = seen[0]

        episodes.append((hdf5_path, mp4_src, prefix, idx, instruction))
    return episodes


def main():
    # 按 task 分组收集 + 各自切 90/10
    train_eps = []
    val_eps = []

    for prefix, task_name in TASKS.items():
        eps = collect_task_episodes(prefix, task_name)
        n = len(eps)
        n_train = int(n * (1 - VAL_RATIO))
        n_val = n - n_train
        train_eps.extend(eps[:n_train])
        val_eps.extend(eps[n_train:])
        print(f"[{prefix}={task_name}] 总共 {n} 个 → train {n_train} + val {n_val}")

    print(f"\n汇总：train {len(train_eps)} + val {len(val_eps)} = {len(train_eps) + len(val_eps)}")

    # 转换 train
    for hdf5_path, mp4_src, prefix, idx, instruction in tqdm(train_eps, desc="train"):
        ep_id = f"{prefix}_{idx:04d}"
        json_dst = OUTPUT_BASE / "annotation" / "train" / f"{ep_id}.json"
        mp4_dst  = OUTPUT_BASE / "videos" / f"{ep_id}.mp4"
        try:
            convert_episode(hdf5_path, mp4_src, mp4_dst, json_dst, ep_id, instruction)
        except Exception as e:
            print(f"  [ERROR] {hdf5_path}: {e}")

    # 转换 val
    for hdf5_path, mp4_src, prefix, idx, instruction in tqdm(val_eps, desc="val"):
        ep_id = f"{prefix}_{idx:04d}"
        json_dst = OUTPUT_BASE / "annotation" / "val" / f"{ep_id}.json"
        mp4_dst  = OUTPUT_BASE / "videos" / f"{ep_id}.mp4"
        try:
            convert_episode(hdf5_path, mp4_src, mp4_dst, json_dst, ep_id, instruction)
        except Exception as e:
            print(f"  [ERROR] {hdf5_path}: {e}")

    # 验证
    print(f"\n完成！输出目录：{OUTPUT_BASE}")
    train_files = sorted((OUTPUT_BASE / 'annotation' / 'train').glob('*.json'))
    val_files = sorted((OUTPUT_BASE / 'annotation' / 'val').glob('*.json'))

    def count_by_prefix(files):
        c = {}
        for f in files:
            p = f.stem.split('_')[0]
            c[p] = c.get(p, 0) + 1
        return c

    print(f"  annotation/train/: {len(train_files)} 个 — {count_by_prefix(train_files)}")
    print(f"  annotation/val/:   {len(val_files)} 个 — {count_by_prefix(val_files)}")
    print(f"  videos/:           {len(list((OUTPUT_BASE / 'videos').glob('*.mp4')))} 个")


if __name__ == "__main__":
    main()

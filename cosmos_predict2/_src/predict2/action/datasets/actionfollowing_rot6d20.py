# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical ActionFollowingData Rot6D20 loader for Cosmos-Predict2.5.

This is intentionally a separate dataset rather than another branch in the
legacy RoboTwin 14D loader.  Actions are read verbatim from the canonical
LeRobot mirror; no Euler conversion and no legacy x20 scaler are applied.
"""

from __future__ import annotations

import json
import math
import os
import random
from bisect import bisect_right
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils.dataset_utils import Resize_Preprocess, ToTensorVideo

Protocol = Literal["clean", "mix4"]

ACTION_DIM = 20
STATE_DIM = 20
ACTION_FEATURE = "action"
STATE_FEATURE = "observation.state"
HEAD_FEATURE = "observation.images.cam_high"
MAX_SAMPLE_ATTEMPTS = 64

ROBOTWIN_50_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)

MIX4_TARGET_RATIOS = {
    "clean": 4.0,
    "perturbed": 1.0,
    "random_feasible": 1.0,
    "counterfactual_replay": 1.0,
    "exploration": 1.0,
}


def actionfollowing_sources(root: str | Path, protocol: Protocol) -> list[tuple[str, str, str]]:
    root = Path(root)
    if protocol not in ("clean", "mix4"):
        raise ValueError(f"Unsupported ActionFollowingData protocol: {protocol!r}")
    sources: list[tuple[str, str, str]] = []
    for task in ROBOTWIN_50_TASKS:
        sources.append((str(root / "demo_clean_zed2i_visible" / task), "clean", task))
        if protocol == "mix4":
            sources.extend(
                [
                    (str(root / "perturbed" / "pca" / "tasks" / task), "perturbed", task),
                    (str(root / "perturbed" / "raw" / "tasks" / task), "perturbed", task),
                    (
                        str(root / "random_feasible" / "uniform" / "tasks" / task),
                        "random_feasible",
                        task,
                    ),
                    (
                        str(root / "random_feasible" / "weighted" / "tasks" / task),
                        "random_feasible",
                        task,
                    ),
                    (
                        str(root / "counterfactual_replay" / "tasks" / task),
                        "counterfactual_replay",
                        task,
                    ),
                    (str(root / "exploration" / "tasks" / task), "exploration", task),
                ]
            )
    return sources


def _read_episode_rows(root: Path) -> list[dict[str, Any]]:
    candidates = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    direct = root / "meta" / "episodes.parquet"
    if direct.is_file():
        candidates.append(direct)
    if not candidates:
        raise FileNotFoundError(f"No LeRobot episode metadata under {root / 'meta'}")
    rows = [row for path in candidates for row in pq.read_table(path).to_pylist()]
    return sorted(rows, key=lambda row: int(row["episode_index"]))


class ActionFollowingRot6D20Dataset(Dataset):
    """Full-50 canonical dataset with deterministic chunk-level protocol sampling."""

    def __init__(
        self,
        root: str,
        protocol: Protocol,
        chunk_length: int = 32,
        video_size: tuple[int, int] | list[int] = (256, 320),
        mode: str = "train",
        seed: int = 20260717,
        audit_num_samples: int = 10000,
        audit_max_abs_error: float = 0.02,
        source_cache_size: int = 4,
        video_reader_cache_size: int = 4,
        device_payload: bool = True,
        gripper_rescale_factor: float = 1.0,
        num_action_per_chunk: int | None = None,
        fps_downsample_ratio: int = 1,
    ) -> None:
        super().__init__()
        if chunk_length != 32:
            raise ValueError(f"ActionFollowingData protocol requires chunk_length=32, got {chunk_length}")
        if num_action_per_chunk is not None and int(num_action_per_chunk) != chunk_length:
            raise ValueError(
                "ActionFollowingData requires num_action_per_chunk to match "
                f"chunk_length={chunk_length}, got {num_action_per_chunk}"
            )
        if float(gripper_rescale_factor) != 1.0:
            raise ValueError(
                "Canonical Rot6D20 actions must not rescale grippers; "
                f"got gripper_rescale_factor={gripper_rescale_factor}"
            )
        if int(fps_downsample_ratio) != 1:
            raise ValueError(
                "Canonical ActionFollowingData timestamps must not be downsampled; "
                f"got fps_downsample_ratio={fps_downsample_ratio}"
            )
        if mode != "train":
            raise ValueError("The frozen baseline loader currently exposes the full training split only.")
        self.root = Path(root)
        self.protocol = protocol
        self.chunk_length = int(chunk_length)
        self.sequence_length = self.chunk_length + 1
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.mode = mode
        self.seed = int(seed)
        self.device_payload = bool(device_payload)
        self.source_cache_size = int(source_cache_size)
        self.video_reader_cache_size = int(video_reader_cache_size)

        self.source_specs = actionfollowing_sources(root, protocol)
        self.family_by_source = [family for _, family, _ in self.source_specs]
        self.task_by_source = [task for _, _, task in self.source_specs]
        self.source_infos: list[dict[str, Any]] = []
        self.episodes_by_source: list[dict[int, dict[str, Any]]] = []
        self.episode_records: list[tuple[int, int, int, int]] = []
        self.record_families: list[str] = []
        self.record_mass_cum_ends: list[float] = []
        self.sample_mass_by_family: dict[str, float] = {}
        self.total_sampling_mass = 0.0
        self.num_virtual_samples = 0

        self._source_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._video_reader_cache: OrderedDict[str, VideoReader] = OrderedDict()
        self._preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(self.video_size)])

        self._build_episode_index()
        self._build_protocol_sampling_index()
        audit = self.audit_sampling(num_samples=audit_num_samples)
        if audit["max_abs_error"] > float(audit_max_abs_error):
            raise RuntimeError(
                f"ActionFollowingData sampling audit failed: {audit['max_abs_error']:.6f} "
                f"> {audit_max_abs_error:.6f}; audit={audit}"
            )
        print(f"ActionFollowingData family effective counts: {self.family_effective_counts}")
        print(f"ActionFollowingData family per-chunk weights: {self.family_per_chunk_weights}")
        print(f"ActionFollowingData sampling audit: {audit}")

    @staticmethod
    def _metadata_for_root(root: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        source_root = Path(root)
        info_path = source_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        info = json.loads(info_path.read_text())
        features = info.get("features", {})
        for key, expected_width in ((ACTION_FEATURE, ACTION_DIM), (STATE_FEATURE, STATE_DIM)):
            shape = features.get(key, {}).get("shape") or []
            actual_width = int(shape[-1]) if shape else None
            if actual_width != expected_width:
                raise ValueError(f"{source_root}: expected {key} width {expected_width}, got {actual_width}")
        if HEAD_FEATURE not in features:
            raise ValueError(f"{source_root}: missing required head camera {HEAD_FEATURE}")
        return info, _read_episode_rows(source_root)

    def _build_episode_index(self) -> None:
        roots = [source_root for source_root, _, _ in self.source_specs]
        missing = [root for root in roots if not Path(root).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} ActionFollowingData roots; first={missing[0]}")
        with ThreadPoolExecutor(max_workers=min(64, len(roots))) as executor:
            metadata = list(executor.map(self._metadata_for_root, roots))

        for source_index, (info, episodes) in enumerate(metadata):
            self.source_infos.append(info)
            episodes_by_id = {int(episode["episode_index"]): episode for episode in episodes}
            if len(episodes_by_id) != len(episodes):
                raise ValueError(f"Duplicate episode_index values in {self.source_specs[source_index][0]}")
            self.episodes_by_source.append(episodes_by_id)
            family = self.family_by_source[source_index]
            for episode in episodes:
                episode_id = int(episode["episode_index"])
                sample_start = int(episode.get("dataset_from_index", episode.get("data/from", 0)))
                length = int(episode["length"])
                if family == "perturbed":
                    valid_len = 1
                else:
                    valid_len = max(1, length - self.chunk_length + 1)
                self.episode_records.append((source_index, sample_start, valid_len, episode_id))
                self.num_virtual_samples += valid_len

    def _build_protocol_sampling_index(self) -> None:
        family_counts: Counter[str] = Counter()
        for source_index, _start, valid_len, _episode_id in self.episode_records:
            family_counts[self.family_by_source[source_index]] += int(valid_len)
        self.family_effective_counts = dict(sorted(family_counts.items()))

        raw_targets = {"clean": 1.0} if self.protocol == "clean" else MIX4_TARGET_RATIOS
        absent = sorted(set(raw_targets) - set(family_counts))
        unexpected = sorted(set(family_counts) - set(raw_targets))
        if absent or unexpected:
            raise ValueError(f"Protocol family mismatch: absent={absent}, unexpected={unexpected}")
        target_total = float(sum(raw_targets.values()))
        self.target_family_probabilities = {
            family: float(weight / target_total) for family, weight in raw_targets.items()
        }
        self.sample_mass_by_family = {
            family: self.target_family_probabilities[family] / float(family_counts[family]) for family in family_counts
        }

        running_mass = 0.0
        for source_index, _start, valid_len, _episode_id in self.episode_records:
            family = self.family_by_source[source_index]
            running_mass += float(valid_len) * self.sample_mass_by_family[family]
            self.record_mass_cum_ends.append(running_mass)
            self.record_families.append(family)
        self.total_sampling_mass = running_mass
        if not math.isclose(running_mass, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"Protocol sampling mass must sum to 1, got {running_mass}")
        normalizer = self.sample_mass_by_family.get("perturbed", next(iter(self.sample_mass_by_family.values())))
        self.family_per_chunk_weights = {
            family: mass / normalizer for family, mass in sorted(self.sample_mass_by_family.items())
        }

        self.family_record_indices: dict[str, list[int]] = {
            family: [] for family in self.target_family_probabilities
        }
        self.family_record_cum_ends: dict[str, list[int]] = {
            family: [] for family in self.target_family_probabilities
        }
        family_running_counts: Counter[str] = Counter()
        for record_index, (source_index, _start, valid_len, _episode_id) in enumerate(self.episode_records):
            family = self.family_by_source[source_index]
            family_running_counts[family] += int(valid_len)
            self.family_record_indices[family].append(record_index)
            self.family_record_cum_ends[family].append(family_running_counts[family])

    @staticmethod
    def _coprime_multiplier(length: int, rng: random.Random) -> int:
        if length <= 1:
            return 1
        candidate = rng.randrange(1, length)
        while math.gcd(candidate, length) != 1:
            candidate = (candidate + 1) % length or 1
        return candidate

    def _sample_quantile(self, index: int, length: int, seed: int, epoch: int = 0) -> float:
        rng = random.Random(int(seed) + (int(epoch) + 1) * 1_000_003 + int(length) * 9_176)
        multiplier = self._coprime_multiplier(length, rng)
        offset = rng.randrange(length) if length > 1 else 0
        shift = rng.random()
        rank = (multiplier * (int(index) % length) + offset) % length
        return (float(rank) + shift) / float(length)

    def _weighted_record_offset(
        self,
        index: int,
        *,
        length: int | None = None,
        seed: int | None = None,
    ) -> tuple[int, int]:
        length = len(self) if length is None else int(length)
        quantile = self._sample_quantile(index, length, self.seed if seed is None else int(seed))
        target_mass = quantile * self.total_sampling_mass
        record_index = min(bisect_right(self.record_mass_cum_ends, target_mass), len(self.episode_records) - 1)
        previous_mass = 0.0 if record_index == 0 else self.record_mass_cum_ends[record_index - 1]
        family = self.record_families[record_index]
        valid_len = int(self.episode_records[record_index][2])
        offset = min(int((target_mass - previous_mass) / self.sample_mass_by_family[family]), valid_len - 1)
        return record_index, max(0, offset)

    def audit_sampling(self, num_samples: int = 10000, seed: int = 20260717) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        for index in range(int(num_samples)):
            record_index, _ = self._weighted_record_offset(index, length=num_samples, seed=seed)
            counts[self.record_families[record_index]] += 1
        observed = {family: counts[family] / float(num_samples) for family in self.target_family_probabilities}
        max_abs_error = max(
            abs(observed[family] - target) for family, target in self.target_family_probabilities.items()
        )
        return {
            "num_samples": int(num_samples),
            "seed": int(seed),
            "family_counts": dict(counts),
            "observed_family_probabilities": observed,
            "target_family_probabilities": self.target_family_probabilities,
            "max_abs_error": float(max_abs_error),
        }

    def __len__(self) -> int:
        return self.num_virtual_samples

    def _load_source(self, source_index: int) -> dict[str, np.ndarray]:
        cached = self._source_cache.get(source_index)
        if cached is not None:
            self._source_cache.move_to_end(source_index)
            return cached

        source_root = Path(self.source_specs[source_index][0])
        parquet_files = sorted((source_root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No LeRobot parquet under {source_root / 'data'}")
        columns = ["index", "episode_index", "frame_index", "timestamp", ACTION_FEATURE]
        tables = [pq.read_table(path, columns=columns) for path in parquet_files]
        arrays = {
            "index": np.concatenate([table["index"].to_numpy() for table in tables]).astype(np.int64),
            "episode_index": np.concatenate([table["episode_index"].to_numpy() for table in tables]).astype(np.int64),
            "frame_index": np.concatenate([table["frame_index"].to_numpy() for table in tables]).astype(np.int64),
            "timestamp": np.concatenate([table["timestamp"].to_numpy() for table in tables]).astype(np.float64),
            "action": np.concatenate(
                [np.asarray(table[ACTION_FEATURE].to_pylist(), dtype=np.float32) for table in tables], axis=0
            ),
        }
        order = np.argsort(arrays["index"], kind="stable")
        arrays = {key: value[order] for key, value in arrays.items()}
        if arrays["action"].ndim != 2 or arrays["action"].shape[1] != ACTION_DIM:
            raise ValueError(f"{source_root}: expected action [N,{ACTION_DIM}], got {arrays['action'].shape}")

        self._source_cache[source_index] = arrays
        self._source_cache.move_to_end(source_index)
        while len(self._source_cache) > self.source_cache_size:
            self._source_cache.popitem(last=False)
        return arrays

    def _video_path(self, source_index: int, episode: dict[str, Any]) -> Path:
        info = self.source_infos[source_index]
        chunk_index = int(
            episode.get(
                f"videos/{HEAD_FEATURE}/chunk_index",
                episode.get(f"videos/{HEAD_FEATURE}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        file_index = int(
            episode.get(
                f"videos/{HEAD_FEATURE}/file_index",
                episode.get(f"videos/{HEAD_FEATURE}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        relative = info["video_path"].format(
            video_key=HEAD_FEATURE,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
            episode_file=file_index,
        )
        source_root = Path(self.source_specs[source_index][0])
        primary = source_root / relative
        if primary.is_file():
            return primary

        remap = os.environ.get("AFD_VIDEO_SYMLINK_PREFIX_REMAP", "")
        old_prefix, separator, new_prefix = remap.partition("=")
        if separator and primary.is_symlink():
            target = Path(os.readlink(primary))
            if target.is_absolute():
                try:
                    remapped = Path(new_prefix) / target.relative_to(old_prefix)
                except ValueError:
                    pass
                else:
                    if remapped.is_file():
                        return remapped

        source_relative = source_root.relative_to(self.root)
        fallback_roots: list[Path] = []
        if self.root.name == "train":
            fallback_roots.append(self.root.parent)
        fallback_roots.extend(
            Path(value)
            for value in os.environ.get("AFD_VIDEO_FALLBACK_ROOTS", "").split(":")
            if value
        )
        for fallback_root in fallback_roots:
            candidate = fallback_root / source_relative / relative
            if candidate.is_file():
                return candidate
        return primary

    def _get_video_reader(self, path: Path) -> VideoReader:
        key = str(path)
        reader = self._video_reader_cache.get(key)
        if reader is None:
            reader = VideoReader(key, ctx=cpu(0), num_threads=2)
            self._video_reader_cache[key] = reader
        self._video_reader_cache.move_to_end(key)
        while len(self._video_reader_cache) > self.video_reader_cache_size:
            self._video_reader_cache.popitem(last=False)
        return reader

    def _decode_head_video(
        self,
        source_index: int,
        episode: dict[str, Any],
        timestamps: np.ndarray,
    ) -> torch.Tensor:
        reader = self._get_video_reader(self._video_path(source_index, episode))
        from_timestamp = float(episode.get(f"videos/{HEAD_FEATURE}/from_timestamp", 0.0))
        fps = float(reader.get_avg_fps())
        frame_indices = np.rint((from_timestamp + timestamps) * fps).astype(np.int64)
        frame_indices = np.clip(frame_indices, 0, len(reader) - 1)
        frames = torch.from_numpy(
            np.ascontiguousarray(reader.get_batch(frame_indices.tolist()).asnumpy().astype(np.uint8))
        ).permute(0, 3, 1, 2)
        frames = self._preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        frames = torch.cat([frames, frames[-1:]], dim=0)
        return frames.permute(1, 0, 2, 3)

    def _device_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.cuda() if self.device_payload else tensor

    def _build_item_from_record(self, record_index: int, frame_offset: int) -> dict[str, Any]:
        source_index, row_start, _valid_len, episode_id = self.episode_records[record_index]
        row_index = int(row_start) + int(frame_offset)
        source = self._load_source(source_index)
        stop = row_index + self.chunk_length
        actions = torch.from_numpy(np.ascontiguousarray(source["action"][row_index:stop])).float()
        timestamps = source["timestamp"][row_index:stop]
        if tuple(actions.shape) != (self.chunk_length, ACTION_DIM):
            raise ValueError(f"Expected action ({self.chunk_length},{ACTION_DIM}), got {tuple(actions.shape)}")
        episode = self.episodes_by_source[source_index][episode_id]
        video = self._decode_head_video(source_index, episode, timestamps)

        return {
            "action": actions,
            "video": video,
            "annotation_file": str(self.source_specs[source_index][0]),
            "__key__": f"{self.task_by_source[source_index]}:{episode_id}:{frame_offset}",
            "t5_text_embeddings": self._device_tensor(torch.zeros(512, 1024, dtype=torch.bfloat16)),
            "ai_caption": "",
            "t5_text_mask": self._device_tensor(torch.ones(512, dtype=torch.int64)),
            "fps": int(round(float(self.source_infos[source_index]["fps"]))),
            # Keep the established Predict2.5 action-conditioned conditioner
            # metadata exactly aligned with Dataset_3D.
            "image_size": self._device_tensor(256 * torch.ones(4)),
            "num_frames": self.sequence_length,
            "padding_mask": self._device_tensor(torch.zeros(1, 256, 256)),
            "family": self.family_by_source[source_index],
            "task_name": self.task_by_source[source_index],
        }

    def _build_item(self, index: int) -> dict[str, Any]:
        record_index, frame_offset = self._weighted_record_offset(index)
        return self._build_item_from_record(record_index, frame_offset)

    def _family_retry_record_offset(self, family: str, index: int, attempt: int) -> tuple[int, int]:
        """Choose a deterministic replacement chunk without changing its sampled family.

        Some mirrored LeRobot episode rows point at absent video assets. A bounded
        replacement must preserve the protocol's family probability instead of
        silently turning a missing enhanced sample into a clean sample.
        """

        cumulative = self.family_record_cum_ends[family]
        total = cumulative[-1]
        family_salt = sum((position + 1) * ord(char) for position, char in enumerate(family))
        rng = random.Random(self.seed + int(index) * 1_000_003 + family_salt * 9_176)
        multiplier = self._coprime_multiplier(total, rng)
        offset = rng.randrange(total) if total > 1 else 0
        sample_rank = (multiplier * int(attempt) + offset) % total
        family_record_position = bisect_right(cumulative, sample_rank)
        previous_end = 0 if family_record_position == 0 else cumulative[family_record_position - 1]
        record_index = self.family_record_indices[family][family_record_position]
        return record_index, sample_rank - previous_end

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, frame_offset = self._weighted_record_offset(index)
        target_family = self.record_families[record_index]
        last_error: Exception | None = None
        for attempt in range(MAX_SAMPLE_ATTEMPTS):
            try:
                return self._build_item_from_record(record_index, frame_offset)
            except Exception as error:  # noqa: BLE001 - bounded resample for corrupt video frames
                last_error = error
                record_index, frame_offset = self._family_retry_record_offset(
                    target_family,
                    int(index),
                    attempt + 1,
                )
        raise RuntimeError(
            "Failed to load ActionFollowingData "
            f"{target_family} sample after {MAX_SAMPLE_ATTEMPTS} attempts: {last_error}"
        )


def build_actionfollowing_dataloader(
    *,
    dataloaders: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DataLoader:
    """Build the canonical loader while discarding one known parent-config residue.

    The official Predict2.5 base experiment contributes a ``dataloaders`` map
    for joint image/video pretraining. Hydra recursively merges that map even
    after ``data_train`` is overridden with this single-dataset loader. All
    remaining keys are forwarded to PyTorch so any other incompatible residue
    still fails loudly.
    """

    del dataloaders
    return DataLoader(**kwargs)


__all__ = [
    "ACTION_DIM",
    "MIX4_TARGET_RATIOS",
    "ROBOTWIN_50_TASKS",
    "ActionFollowingRot6D20Dataset",
    "actionfollowing_sources",
    "build_actionfollowing_dataloader",
]

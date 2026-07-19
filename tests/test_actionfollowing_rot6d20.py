import pytest
import torch

from cosmos_predict2._src.predict2.action.datasets.actionfollowing_rot6d20 import (
    ACTION_DIM,
    ROBOTWIN_50_TASKS,
    ActionFollowingRot6D20Dataset,
    actionfollowing_sources,
    build_actionfollowing_dataloader,
)


def test_full50_source_inventory() -> None:
    assert ACTION_DIM == 20
    assert len(ROBOTWIN_50_TASKS) == 50
    assert len(actionfollowing_sources("/dataset", "clean")) == 50
    assert len(actionfollowing_sources("/dataset", "mix4")) == 350


def test_mix4_sampling_matches_cosmos3_and_protocol() -> None:
    counts = {
        "clean": 475122,
        "perturbed": 1000000,
        "random_feasible": 2700000,
        "counterfactual_replay": 957810,
        "exploration": 239844,
    }
    dataset = ActionFollowingRot6D20Dataset.__new__(ActionFollowingRot6D20Dataset)
    dataset.protocol = "mix4"
    dataset.seed = 20260717
    dataset.family_by_source = list(counts)
    dataset.episode_records = [
        (source_index, 0, count, source_index) for source_index, count in enumerate(counts.values())
    ]
    dataset.num_virtual_samples = sum(counts.values())
    dataset.record_families = []
    dataset.record_mass_cum_ends = []
    dataset.sample_mass_by_family = {}
    dataset.total_sampling_mass = 0.0

    dataset._build_protocol_sampling_index()
    audit = dataset.audit_sampling(num_samples=100000, seed=20260717)

    assert audit["max_abs_error"] <= 0.001
    assert audit["target_family_probabilities"] == {
        "clean": 0.5,
        "perturbed": 0.125,
        "random_feasible": 0.125,
        "counterfactual_replay": 0.125,
        "exploration": 0.125,
    }


def test_inherited_robotwin_dataset_kwargs_are_canonical_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ActionFollowingRot6D20Dataset, "_build_episode_index", lambda self: None)

    def fake_build_protocol_sampling_index(dataset: ActionFollowingRot6D20Dataset) -> None:
        dataset.family_effective_counts = {}
        dataset.family_per_chunk_weights = {}

    monkeypatch.setattr(
        ActionFollowingRot6D20Dataset,
        "_build_protocol_sampling_index",
        fake_build_protocol_sampling_index,
    )
    monkeypatch.setattr(
        ActionFollowingRot6D20Dataset,
        "audit_sampling",
        lambda self, num_samples: {"max_abs_error": 0.0},
    )

    dataset = ActionFollowingRot6D20Dataset(
        root="/dataset",
        protocol="mix4",
        gripper_rescale_factor=1,
        num_action_per_chunk=32,
        fps_downsample_ratio=1,
    )
    assert dataset.chunk_length == 32

    with pytest.raises(ValueError, match="num_action_per_chunk"):
        ActionFollowingRot6D20Dataset(
            root="/dataset",
            protocol="mix4",
            num_action_per_chunk=12,
        )


def test_parent_joint_dataloaders_residue_is_discarded() -> None:
    loader = build_actionfollowing_dataloader(
        dataloaders={"image_data": object(), "video_data": object()},
        dataset=torch.utils.data.TensorDataset(torch.arange(4)),
        batch_size=2,
        num_workers=0,
    )
    assert len(loader) == 2


def test_missing_video_retries_preserve_sampled_family(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = ActionFollowingRot6D20Dataset.__new__(ActionFollowingRot6D20Dataset)
    dataset.seed = 20260717
    dataset.episode_records = [(0, 0, 1, 0), (1, 0, 1, 0), (2, 0, 1, 0)]
    dataset.record_families = ["clean", "perturbed", "perturbed"]
    dataset.family_record_indices = {"clean": [0], "perturbed": [1, 2]}
    dataset.family_record_cum_ends = {"clean": [1], "perturbed": [1, 2]}
    monkeypatch.setattr(dataset, "_weighted_record_offset", lambda index: (1, 0))

    attempted_families: list[str] = []

    def fake_build(record_index: int, frame_offset: int) -> dict[str, str]:
        del frame_offset
        family = dataset.record_families[record_index]
        attempted_families.append(family)
        if len(attempted_families) < 3:
            raise FileNotFoundError("missing mirrored video")
        return {"family": family}

    monkeypatch.setattr(dataset, "_build_item_from_record", fake_build)

    assert dataset[7]["family"] == "perturbed"
    assert set(attempted_families) == {"perturbed"}


def test_split_video_path_falls_back_to_full_root(tmp_path) -> None:
    split_root = tmp_path / "train"
    relative = "videos/observation.images.cam_high/chunk-000/file-000003.mp4"
    full_video = tmp_path / "exploration" / "tasks" / "turn_switch" / relative
    full_video.parent.mkdir(parents=True)
    full_video.touch()

    dataset = ActionFollowingRot6D20Dataset.__new__(ActionFollowingRot6D20Dataset)
    dataset.root = split_root
    dataset.source_specs = [(str(split_root / "exploration/tasks/turn_switch"), "exploration", "turn_switch")]
    dataset.source_infos = [
        {
            "video_path": (
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4"
            )
        }
    ]
    episode = {
        "videos/observation.images.cam_high/chunk_index": 0,
        "videos/observation.images.cam_high/file_index": 3,
    }

    assert dataset._video_path(0, episode) == full_video


def test_broken_absolute_video_symlink_uses_mount_prefix_remap(tmp_path, monkeypatch) -> None:
    split_root = tmp_path / "train"
    relative = "videos/observation.images.cam_high/chunk-000/file-000003.mp4"
    source_root = split_root / "exploration/tasks/turn_switch"
    broken_link = source_root / relative
    broken_link.parent.mkdir(parents=True)
    broken_target = "/old/data/ActionFollowingData/enhanced/sample/video.mp4"
    broken_link.symlink_to(broken_target)
    remapped = tmp_path / "mounted" / "ActionFollowingData/enhanced/sample/video.mp4"
    remapped.parent.mkdir(parents=True)
    remapped.touch()
    monkeypatch.setenv("AFD_VIDEO_SYMLINK_PREFIX_REMAP", f"/old/data={tmp_path / 'mounted'}")

    dataset = ActionFollowingRot6D20Dataset.__new__(ActionFollowingRot6D20Dataset)
    dataset.root = split_root
    dataset.source_specs = [(str(source_root), "exploration", "turn_switch")]
    dataset.source_infos = [
        {"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4"}
    ]
    episode = {
        "videos/observation.images.cam_high/chunk_index": 0,
        "videos/observation.images.cam_high/file_index": 3,
    }

    assert dataset._video_path(0, episode) == remapped

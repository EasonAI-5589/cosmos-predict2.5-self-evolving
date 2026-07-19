# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predict2.5 full-50 ActionFollowingData Rot6D20 training registration."""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.configs.action_conditioned.data import get_sampler
from cosmos_predict2._src.predict2.action.configs.action_conditioned.experiment.exp_2B_action_conditioned_rectify_flow import (
    AC_ROBOTWIN_14D_RECTIFIED_FLOW_2B_8GPU,
)
from cosmos_predict2._src.predict2.action.datasets.actionfollowing_rot6d20 import (
    ActionFollowingRot6D20Dataset,
    build_actionfollowing_dataloader,
)

actionfollowing_rot6d20_dataset = L(ActionFollowingRot6D20Dataset)(
    root="${oc.env:AFD_ROOT}",
    protocol="${oc.env:AFD_PROTOCOL}",
    chunk_length=32,
    video_size=[256, 320],
    mode="train",
    seed=20260717,
    audit_num_samples=10000,
    audit_max_abs_error=0.02,
    source_cache_size=4,
    video_reader_cache_size=4,
    device_payload=False,
)

actionfollowing_rot6d20_dataloader = L(build_actionfollowing_dataloader)(
    dataset=actionfollowing_rot6d20_dataset,
    sampler=L(get_sampler)(dataset=actionfollowing_rot6d20_dataset),
    batch_size=2,
    num_workers=4,
    prefetch_factor=1,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
)

ConfigStore.instance().store(
    group="data_train",
    package="dataloader_train",
    name="actionfollowing_rot6d20",
    node=actionfollowing_rot6d20_dataloader,
)


COSMOS_PREDICT25_AFD_ROT6D20 = LazyDict(
    dict(
        defaults=[
            f"/experiment/{AC_ROBOTWIN_14D_RECTIFIED_FLOW_2B_8GPU['job']['name']}",
            {"override /data_train": "actionfollowing_rot6d20"},
            {"override /data_val": "mock"},
        ],
        upload_reproducible_setup=False,
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="actionfollowing_full50",
            name="cosmos_predict25_afd_full50_rot6d20_a32_bs16_40k",
            wandb_mode="disabled",
        ),
        checkpoint=dict(
            load_training_state=False,
            strict_resume=False,
            save_iter=1000,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        model=dict(
            config=dict(
                state_t=1 + 32 // 4,
                enable_value_frame=False,
                net=dict(
                    action_dim=20,
                    num_action_per_chunk=32,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        trainer=dict(
            max_iter=40000,
            logging_iter=10,
            run_validation=False,
            seed=20260717,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=1_000_000_000, save_s3=False),
                every_n_sample_ema=dict(every_n=1_000_000_000, save_s3=False),
            ),
        ),
        dataloader_train=dict(
            batch_size=2,
            num_workers=4,
            prefetch_factor=1,
            sampler=dict(
                dataset=dict(
                    gripper_rescale_factor=1,
                    num_action_per_chunk=32,
                    fps_downsample_ratio=1,
                    video_size=[256, 320],
                ),
            ),
            dataset=dict(
                gripper_rescale_factor=1,
                num_action_per_chunk=32,
                fps_downsample_ratio=1,
                video_size=[256, 320],
            ),
        ),
    ),
    flags={"allow_objects": True},
)


ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="cosmos_predict25_actionfollowing_rot6d20",
    node=COSMOS_PREDICT25_AFD_ROT6D20,
)

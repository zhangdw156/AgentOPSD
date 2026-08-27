from __future__ import annotations

# pyright: reportAttributeAccessIssue=false
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.utils import adjust_batch
from verl import DataProto
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.trajectory_grpo import (
    NATIVE_TRAJECTORY_GRPO_CONFIG,
    validate_trajectory_grpo_config,
)


class _StopAfterActorUpdate(Exception):
    pass


class _ActorRolloutStub:
    world_size = 1

    def compute_log_prob(self, batch):
        return DataProto.from_dict(
            tensors={
                "entropys": torch.zeros_like(
                    batch.batch["responses"],
                    dtype=torch.float32,
                ),
                "old_log_probs": torch.zeros_like(
                    batch.batch["responses"],
                    dtype=torch.float32,
                ),
            }
        )

    def update_actor(self, batch):
        assert "advantages" in batch.batch
        assert "returns" in batch.batch
        assert torch.count_nonzero(batch.batch["advantages"]).item() > 0
        torch.testing.assert_close(
            batch.batch["returns"],
            batch.batch["advantages"],
        )
        raise _StopAfterActorUpdate


def _fit_config():
    return OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": "console",
                "val_before_train": False,
                "total_epochs": 1,
                "balance_batch": False,
                "critic_warmup": 0,
                "test_freq": 0,
                "save_freq": 0,
                "rollout_data_dir": None,
            },
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
                "use_kl_in_reward": False,
                "use_pf_ppo": False,
                "pf_ppo": {
                    "reweight_method": "pow",
                    "weight_pow": 2.0,
                },
                "gigpo": {
                    "step_advantage_w": 1.0,
                    "mode": "mean_std_norm",
                    "enable_similarity": False,
                    "similarity_thresh": 0.95,
                },
                "trajectory_grpo": dict(
                    NATIVE_TRAJECTORY_GRPO_CONFIG
                ),
            },
            "actor_rollout_ref": {
                "actor": {
                    "loss_agg_mode": "token-mean",
                    "use_invalid_action_penalty": False,
                },
                "rollout": {
                    "n": 2,
                    "multi_turn": {"enable": False},
                },
            },
            "reward_model": {
                "launch_reward_fn_async": False,
            },
        }
    )


def _rollout_batch():
    responses = torch.tensor([[11, 12], [21, 22]])
    prompts = torch.tensor([[1, 2], [3, 4]])
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "position_ids": torch.arange(4).repeat(2, 1),
        },
        non_tensors={
            "uid": np.asarray(["group", "group"], dtype=object),
            "traj_uid": np.asarray(
                ["traj-0", "traj-1"],
                dtype=object,
            ),
        },
        meta_info={"temperature": 1.0},
    )


def test_native_step_row_defaults_are_valid():
    validate_trajectory_grpo_config(
        NATIVE_TRAJECTORY_GRPO_CONFIG
    )


def test_native_step_row_adjustment_does_not_add_trajectory_weights():
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
            "actor_rollout_ref": {
                "rollout": {
                    "log_prob_micro_batch_size_per_gpu": 2,
                },
                "actor": {
                    "use_kl_loss": False,
                    "ppo_mini_batch_size": 2,
                    "ppo_micro_batch_size_per_gpu": 2,
                },
                "ref": {
                    "log_prob_micro_batch_size_per_gpu": 2,
                },
            },
            "algorithm": {
                "use_kl_in_reward": False,
                "trajectory_grpo": dict(
                    NATIVE_TRAJECTORY_GRPO_CONFIG
                ),
            },
        }
    )
    data = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1], [2]])},
        non_tensors={
            "uid": np.asarray(["u", "u"], dtype=object),
            "traj_uid": np.asarray(["a", "b"], dtype=object),
        },
    )

    adjusted = adjust_batch(config, data)

    assert adjusted is data
    assert "row_weights" not in adjusted.batch
    assert "trajectory_id" not in adjusted.batch


def test_ray_fit_consumes_step_row_before_actor_update(monkeypatch):
    trainer = object.__new__(ray_trainer.RayPPOTrainer)
    trainer.config = _fit_config()
    trainer.val_reward_fn = None
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "position_ids": torch.arange(2).unsqueeze(0),
            "raw_prompt_ids": np.asarray([[1, 2]], dtype=object),
            "data_source": np.asarray(["alfworld"], dtype=object),
        }
    ]
    trainer.traj_collector = types.SimpleNamespace(
        multi_turn_loop=lambda **_kwargs: _rollout_batch()
    )
    trainer.actor_rollout_wg = _ActorRolloutStub()
    trainer.envs = object()
    trainer.reward_fn = object()
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.ref_in_actor = False
    trainer._load_checkpoint = types.MethodType(
        lambda _self: None,
        trainer,
    )

    tracking_stub = types.ModuleType("verl.utils.tracking")
    tracking_stub.Tracking = type(
        "Tracking",
        (),
        {"__init__": lambda self, **_kwargs: None},
    )
    monkeypatch.setattr(
        ray_trainer,
        "adjust_batch",
        lambda _config, batch: batch,
    )
    monkeypatch.setattr(
        ray_trainer,
        "compute_reward",
        lambda _batch, _reward_fn: (
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            {},
        ),
    )
    monkeypatch.setattr(
        ray_trainer,
        "tqdm",
        lambda **_kwargs: types.SimpleNamespace(),
    )

    with patch.dict(sys.modules, {"verl.utils.tracking": tracking_stub}):
        with pytest.raises(_StopAfterActorUpdate):
            trainer.fit()


def test_opsd_keeps_official_credit_after_step_row_guard():
    source = (
        Path(__file__).parents[3]
        / "verl/trainer/ppo/opsd_ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert "compute_opsd_turn_advantage(" in source
    assert "compute_opsd_token_advantage(" in source
    assert "NATIVE_TRAJECTORY_GRPO_CONFIG" in source
    assert "AgentOPSD preserves its official credit" in source
    assert "self.actor_rollout_wg.update_actor(" in source
    assert "def _compute_teacher_log_probs(" in source

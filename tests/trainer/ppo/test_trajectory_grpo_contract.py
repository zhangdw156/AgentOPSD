from __future__ import annotations

# pyright: reportAttributeAccessIssue=false
import sys
import types
import inspect
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.utils import adjust_batch
from verl import DataProto
from verl.trainer.ppo import opsd_ray_trainer, ray_trainer
from verl.trainer.ppo.opsd_ray_trainer import OPSDRayTrainer
from verl.trainer.ppo.trajectory_grpo import (
    NATIVE_TRAJECTORY_GRPO_CONFIG,
    resolve_trajectory_grpo_config,
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
    assert resolve_trajectory_grpo_config({}) == (
        NATIVE_TRAJECTORY_GRPO_CONFIG
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("scheduler", "trajectory"),
        ("reducer", "trajectory_mean"),
        ("advantage", "trajectory"),
        ("penalty", "trajectory"),
        ("filter", "penalty_aware"),
    ],
)
def test_native_contract_rejects_each_non_native_value(name, value):
    config = dict(NATIVE_TRAJECTORY_GRPO_CONFIG)
    config[name] = value

    with pytest.raises(
        ValueError,
        match=rf"trajectory_grpo\.{name}.*requires",
    ):
        validate_trajectory_grpo_config(config)


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


def test_native_adjustment_preserves_copy_padding_and_traj_uid():
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
            "algorithm": {"use_kl_in_reward": False},
        }
    )
    data = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1], [2], [3]])},
        non_tensors={
            "traj_uid": np.asarray(["a", "b", "c"], dtype=object),
        },
    )

    adjusted = adjust_batch(config, data, mode="copy")

    assert len(adjusted) == 4
    assert adjusted.non_tensor_batch["traj_uid"][:3].tolist() == [
        "a",
        "b",
        "c",
    ]
    assert adjusted.non_tensor_batch["traj_uid"][3] in {"a", "b", "c"}
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


def test_opsd_inherits_fit_and_base_hook_order():
    assert OPSDRayTrainer.fit is ray_trainer.RayPPOTrainer.fit

    source = inspect.getsource(ray_trainer.RayPPOTrainer.fit)
    old_log_prob = source.index(
        'with _timer("old_log_prob", timing_raw):'
    )
    prepare = source.index(
        "batch = self._prepare_advantage_inputs("
    )
    reference = source.index("if self.use_reference_policy:")
    compute = source.index("batch = compute_advantage(")
    postprocess = source.index(
        "batch = self._postprocess_advantages("
    )
    critic = source.index("# update critic")

    assert old_log_prob < prepare < reference
    assert compute < postprocess < critic


def test_opsd_validation_requires_grpo():
    trainer = object.__new__(OPSDRayTrainer)
    trainer.config = OmegaConf.create(
        {"algorithm": {"adv_estimator": "remax"}}
    )

    with pytest.raises(
        ValueError,
        match="AgentOPSD requires algorithm.adv_estimator=grpo",
    ):
        trainer._validate_config()


def test_opsd_postprocess_replaces_advantages_but_preserves_returns(
    monkeypatch,
):
    trainer = object.__new__(OPSDRayTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"opsd": {}},
            "trainer": {"test_freq": 0},
        }
    )
    trainer.global_steps = 1
    trainer.opsd_v0_prior = 0.5
    trainer.opsd_granularity = "token"
    trainer.opsd_use_anchoring = False
    trainer.opsd_use_group_norm = False
    trainer.opsd_center_delta = False
    trainer.opsd_belief_decay_gamma = 1.0
    trainer.opsd_tether_seq = False
    trainer.opsd_tether_lambda = 0.5
    trainer.opsd_tether_band = 0.2
    trainer.opsd_belief_mult = False
    trainer.opsd_mult_lambda_init = 1.0
    trainer.opsd_mult_warmdown_steps = -1
    trainer.opsd_mult_band = 0.2
    trainer.opsd_belief_gate = True
    trainer.opsd_mult_signed = True
    trainer.opsd_signed = False

    original_returns = torch.tensor([[7.0, 0.0]])
    batch = DataProto.from_dict(
        tensors={
            "advantages": torch.tensor([[1.0, 0.0]]),
            "returns": original_returns.clone(),
            "old_log_probs": torch.tensor([[0.1, 0.0]]),
            "teacher_log_probs": torch.tensor([[0.3, 0.0]]),
            "response_mask": torch.tensor([[1, 0]]),
            "token_level_rewards": torch.tensor([[1.0, 0.0]]),
        },
        non_tensors={
            "uid": np.asarray(["group"], dtype=object),
            "traj_uid": np.asarray(["traj"], dtype=object),
        },
    )
    replacement = torch.tensor([[9.0, 0.0]])
    monkeypatch.setattr(
        opsd_ray_trainer,
        "compute_opsd_token_advantage",
        lambda **_kwargs: replacement,
    )

    result = trainer._postprocess_advantages(batch, {})

    assert result is batch
    torch.testing.assert_close(result.batch["advantages"], replacement)
    torch.testing.assert_close(
        result.batch["returns"],
        original_returns,
    )

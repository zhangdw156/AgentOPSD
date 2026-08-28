"""AgentOPSD trainer hooks and privileged teacher-batch construction."""

import json
import os

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.opsd_utils import (
    SkillProvider,
    compute_group_mean_v0,
    compute_opsd_token_advantage,
    compute_opsd_turn_advantage,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
)
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import masked_mean


def build_teacher_batch(
    batch: DataProto,
    skill_provider: SkillProvider,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "error",
    skill_position: str = "legacy",
):
    """Build the skill-conditioned batch used for the teacher forward pass."""
    del truncation
    batch_size = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)
    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    for index in range(batch_size):
        original_input_ids = batch.batch["input_ids"][index]
        original_attention_mask = batch.batch["attention_mask"][index]
        prompt_length = original_input_ids.size(0) - response_length
        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = valid_start[0].item() if len(valid_start) > 0 else 0
        prompt_text = tokenizer.decode(
            prompt_ids[valid_start:],
            skip_special_tokens=False,
        )

        gamefile = batch.non_tensor_batch.get("gamefile")
        data_source = batch.non_tensor_batch.get("data_source")
        if gamefile is not None:
            value = gamefile[index]
            if value is not None:
                skill_text = skill_provider.get_privileged_info(
                    value if isinstance(value, str) else str(value)
                )
            elif data_source is not None:
                value = data_source[index]
                skill_text = (
                    skill_provider.get_privileged_info_from_data_source(
                        value if isinstance(value, str) else str(value),
                        prompt_text,
                    )
                )
            else:
                skill_text = skill_provider.get_privileged_info_from_prompt(
                    prompt_text
                )
        elif data_source is not None:
            value = data_source[index]
            skill_text = skill_provider.get_privileged_info_from_data_source(
                value if isinstance(value, str) else str(value),
                prompt_text,
            )
        else:
            skill_text = skill_provider.get_privileged_info_from_prompt(
                prompt_text
            )

        skill_block = (
            f"[Privileged Skill Information]\n{skill_text}\n\n"
        )
        if skill_position == "user":
            user_marker = "<|im_start|>user\n"
            marker_index = prompt_text.find(user_marker)
            if marker_index != -1:
                insert_at = marker_index + len(user_marker)
                teacher_prompt_text = (
                    prompt_text[:insert_at]
                    + skill_block
                    + prompt_text[insert_at:]
                )
            else:
                teacher_prompt_text = skill_block + prompt_text
        else:
            teacher_prompt_text = skill_block + prompt_text

        teacher_prompt_ids = tokenizer.encode(
            teacher_prompt_text,
            add_special_tokens=False,
        )[-max_prompt_length:]
        teacher_prompt_ids = torch.tensor(
            teacher_prompt_ids,
            dtype=torch.long,
        )
        actual_prompt_length = len(teacher_prompt_ids)
        pad_length = max_prompt_length - actual_prompt_length
        if pad_length > 0:
            teacher_prompt_ids = torch.cat(
                [
                    torch.full(
                        (pad_length,),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                    teacher_prompt_ids,
                ]
            )
            teacher_prompt_mask = torch.cat(
                [
                    torch.zeros(pad_length, dtype=torch.long),
                    torch.ones(actual_prompt_length, dtype=torch.long),
                ]
            )
        else:
            teacher_prompt_mask = torch.ones(
                actual_prompt_length,
                dtype=torch.long,
            )

        response_ids = batch.batch["responses"][index]
        response_mask = original_attention_mask[-response_length:]
        teacher_full_ids = torch.cat(
            [teacher_prompt_ids, response_ids]
        )
        teacher_full_mask = torch.cat(
            [teacher_prompt_mask, response_mask]
        )
        teacher_position_ids = compute_position_id_with_mask(
            teacher_full_mask.unsqueeze(0)
        )[0]
        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    return DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(teacher_input_ids_list),
            "attention_mask": torch.stack(
                teacher_attention_mask_list
            ),
            "position_ids": torch.stack(teacher_position_ids_list),
            "responses": batch.batch["responses"],
        }
    )


class OPSDRayTrainer(RayPPOTrainer):
    """PPO trainer with AgentOPSD privileged credit-assignment hooks."""

    progress_bar_description = "OPSD Training"
    log_rollout_prob_diagnostics = False

    def __init__(
        self,
        *args,
        skill_provider: SkillProvider = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider
        opsd_cfg = self.config.algorithm.get("opsd", {})
        self.opsd_v0_prior = opsd_cfg.get("v0_prior", 0.5)
        self.opsd_skill_position = opsd_cfg.get(
            "skill_position",
            "legacy",
        )
        self.opsd_granularity = opsd_cfg.get("granularity", "token")
        self.opsd_use_anchoring = opsd_cfg.get(
            "use_anchoring",
            False,
        )
        self.opsd_use_group_norm = opsd_cfg.get(
            "use_group_norm",
            False,
        )
        self.opsd_center_delta = opsd_cfg.get("center_delta", False)
        self.opsd_belief_decay_gamma = opsd_cfg.get(
            "belief_decay_gamma",
            1.0,
        )
        self.opsd_tether_seq = opsd_cfg.get("tether_seq", False)
        self.opsd_tether_lambda = opsd_cfg.get("tether_lambda", 0.5)
        self.opsd_tether_band = opsd_cfg.get("tether_band", 0.2)
        self.opsd_belief_mult = opsd_cfg.get("belief_mult", False)
        self.opsd_mult_lambda_init = opsd_cfg.get("mult_lambda", 1.0)
        self.opsd_mult_band = opsd_cfg.get("mult_band", 0.2)
        self.opsd_mult_warmdown_steps = opsd_cfg.get(
            "warmdown_steps",
            -1,
        )
        self.opsd_belief_gate = opsd_cfg.get("belief_gate", True)
        self.opsd_mult_signed = opsd_cfg.get("mult_signed", True)
        self.opsd_signed = opsd_cfg.get("signed", False)

    def _validate_config(self):
        if (
            self.config.algorithm.adv_estimator
            != AdvantageEstimator.GRPO
        ):
            raise ValueError("AgentOPSD requires algorithm.adv_estimator=grpo")
        super()._validate_config()

    def _get_opsd_mult_lambda(self, step: int) -> float:
        warmdown_steps = int(self.opsd_mult_warmdown_steps)
        if warmdown_steps <= 0:
            return float(self.opsd_mult_lambda_init)
        if step >= warmdown_steps:
            return 0.0
        return float(self.opsd_mult_lambda_init) * (
            1.0 - step / warmdown_steps
        )

    def _prepare_advantage_inputs(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
    ) -> DataProto:
        del metrics
        with _timer("teacher_forward", timing_raw):
            batch.batch["teacher_log_probs"] = (
                self._compute_teacher_log_probs(batch)
            )
        return batch

    def _postprocess_advantages(
        self,
        batch: DataProto,
        metrics: dict,
    ) -> DataProto:
        seq_advantages = batch.batch["advantages"]
        student_log_probs = batch.batch["old_log_probs"]
        teacher_log_probs = batch.batch["teacher_log_probs"]
        response_mask = batch.batch["response_mask"]
        token_level_rewards = batch.batch["token_level_rewards"]

        v0_per_traj = None
        if (
            "uid" in batch.non_tensor_batch
            and "traj_uid" in batch.non_tensor_batch
        ):
            v0_per_traj = compute_group_mean_v0(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=batch.non_tensor_batch["uid"],
                traj_index=batch.non_tensor_batch["traj_uid"],
            )

        first_valid = response_mask.long().argmax(dim=-1)
        row_index = torch.arange(
            seq_advantages.size(0),
            device=seq_advantages.device,
        )
        seq_advantage_per_row = seq_advantages[
            row_index,
            first_valid,
        ]
        uid = batch.non_tensor_batch.get("uid")
        current_mult_lambda = self._get_opsd_mult_lambda(
            self.global_steps
        )

        common_kwargs = {
            "token_level_rewards": token_level_rewards,
            "student_log_probs": student_log_probs,
            "teacher_log_probs": teacher_log_probs,
            "response_mask": response_mask,
            "v0_prior": self.opsd_v0_prior,
            "v0_per_traj": v0_per_traj,
            "seq_advantage_per_row": seq_advantage_per_row,
            "uid": uid,
            "use_anchoring": self.opsd_use_anchoring,
            "use_group_norm": self.opsd_use_group_norm,
            "center_delta": self.opsd_center_delta,
            "belief_decay_gamma": self.opsd_belief_decay_gamma,
            "tether_seq": self.opsd_tether_seq,
            "tether_lambda": self.opsd_tether_lambda,
            "tether_band": self.opsd_tether_band,
            "belief_mult": self.opsd_belief_mult,
            "mult_lambda": current_mult_lambda,
            "mult_band": self.opsd_mult_band,
            "belief_gate": self.opsd_belief_gate,
            "mult_signed": self.opsd_mult_signed,
            "signed": self.opsd_signed,
        }
        if self.opsd_granularity == "turn":
            opsd_advantages = compute_opsd_turn_advantage(
                **common_kwargs,
                traj_uid=batch.non_tensor_batch["traj_uid"],
                turn_step=batch.non_tensor_batch["turn_step"],
                episode_rewards=batch.non_tensor_batch.get(
                    "episode_rewards"
                ),
            )
        else:
            opsd_advantages = compute_opsd_token_advantage(
                **common_kwargs
            )
        batch.batch["advantages"] = opsd_advantages

        metrics["opsd/granularity"] = float(
            self.opsd_granularity == "turn"
        )
        metrics["opsd/use_anchoring"] = float(
            self.opsd_use_anchoring
        )
        metrics["opsd/use_group_norm"] = float(
            self.opsd_use_group_norm
        )
        metrics["opsd/center_delta"] = float(self.opsd_center_delta)
        metrics["opsd/belief_decay_gamma"] = float(
            self.opsd_belief_decay_gamma
        )
        metrics["opsd/tether_seq"] = float(self.opsd_tether_seq)
        metrics["opsd/belief_mult"] = float(self.opsd_belief_mult)
        if self.opsd_belief_mult:
            metrics["opsd/mult_lambda"] = float(
                current_mult_lambda
            )
            metrics["opsd/mult_band"] = float(self.opsd_mult_band)
            metrics["opsd/belief_gate"] = float(
                self.opsd_belief_gate
            )
            metrics["opsd/mult_signed"] = float(
                self.opsd_mult_signed
            )
            metrics["opsd/signed"] = float(self.opsd_signed)
        if (
            self.opsd_granularity == "turn"
            and "traj_uid" in batch.non_tensor_batch
        ):
            traj_uids = batch.non_tensor_batch["traj_uid"]
            turn_count = len(traj_uids)
            episode_count = (
                len(set(traj_uids.tolist())) if turn_count else 0
            )
            metrics["opsd/mean_turns_per_episode"] = (
                turn_count / episode_count if episode_count else 0.0
            )

        delta = (
            teacher_log_probs - student_log_probs
        ) * response_mask
        metrics["opsd/teacher_student_gap_mean"] = masked_mean(
            delta,
            response_mask,
        ).item()
        metrics["opsd/teacher_student_gap_std"] = masked_mean(
            delta**2,
            response_mask,
        ).sqrt().item()
        metrics["opsd/v0_mean"] = (
            v0_per_traj.float().mean().item()
            if v0_per_traj is not None
            else float(self.opsd_v0_prior)
        )
        self._maybe_dump_opsd_debug(
            batch,
            seq_advantages,
            student_log_probs,
            teacher_log_probs,
            response_mask,
            delta,
        )
        return batch

    def _maybe_dump_opsd_debug(
        self,
        batch,
        seq_advantages,
        student_log_probs,
        teacher_log_probs,
        response_mask,
        delta,
    ) -> None:
        if not (
            os.environ.get("SAVE_CGTD_DEBUG", "0") == "1"
            and self.config.trainer.test_freq > 0
            and self.global_steps % self.config.trainer.test_freq == 0
        ):
            return

        save_dir = os.environ.get(
            "SAVE_CGTD_DEBUG_DIR",
            "outputs/opsd_debug",
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"step_{self.global_steps}.jsonl",
        )
        batch_size = response_mask.shape[0]
        turn_steps = batch.non_tensor_batch.get(
            "turn_step",
            np.zeros(batch_size, dtype=object),
        )
        traj_uids = batch.non_tensor_batch.get(
            "traj_uid",
            np.array([""] * batch_size, dtype=object),
        )
        episode_rewards = batch.non_tensor_batch.get(
            "episode_rewards",
            np.zeros(batch_size, dtype=object),
        )
        episode_lengths = batch.non_tensor_batch.get(
            "episode_lengths",
            np.zeros(batch_size, dtype=object),
        )
        response_ids = batch.batch["responses"]
        token_advantages = batch.batch["advantages"]

        with open(save_path, "w") as output:
            for index in range(batch_size):
                mask = response_mask[index].bool()
                if mask.sum().item() == 0:
                    continue
                token_ids = response_ids[index][mask].cpu().tolist()
                record = {
                    "global_step": self.global_steps,
                    "sample_idx": index,
                    "turn_step": int(turn_steps[index]),
                    "traj_uid": str(traj_uids[index]),
                    "episode_reward": float(episode_rewards[index]),
                    "episode_length": float(episode_lengths[index]),
                    "seq_advantage": float(
                        seq_advantages[index][mask][0].item()
                    ),
                    "tokens": [
                        self.tokenizer.decode([token_id])
                        for token_id in token_ids
                    ],
                    "token_ids": token_ids,
                    "gaps": delta[index][mask].cpu().tolist(),
                    "teacher_log_probs": teacher_log_probs[index][
                        mask
                    ].cpu().tolist(),
                    "student_log_probs": student_log_probs[index][
                        mask
                    ].cpu().tolist(),
                    "token_advantages": token_advantages[index][
                        mask
                    ].cpu().tolist(),
                }
                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
        print(
            f"[OPSD Debug] Saved per-token gap data to {save_path} "
            f"({batch_size} samples)"
        )

    def _compute_teacher_log_probs(
        self,
        batch: DataProto,
    ) -> torch.Tensor:
        teacher_batch = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
            skill_position=self.opsd_skill_position,
        )
        teacher_output = self.actor_rollout_wg.compute_log_prob(
            teacher_batch
        )
        return teacher_output.batch["old_log_probs"]

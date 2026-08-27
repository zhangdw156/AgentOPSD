#!/usr/bin/env bash
set -euo pipefail

export MODEL_ROOT="${MODEL_ROOT:-/data/zhangdw12/models}"
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export VERL_AGENT_DATA_ROOT="${VERL_AGENT_DATA_ROOT:-${HOME}/data/verl-agent}"
export MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-${VERL_AGENT_DATA_ROOT}/text/train.parquet}"
export VAL_FILE="${VAL_FILE:-${VERL_AGENT_DATA_ROOT}/text/test.parquet}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
VALIDATION_CONCURRENCY="${VALIDATION_CONCURRENCY:-128}"

cd "${REPO_ROOT}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
TRAINER_MODULE="verl.trainer.main_opsd"
DATA_PREP_ARGS=("--mode" "text" "--local_dir" "${VERL_AGENT_DATA_ROOT}" "--train_data_size" "16" "--val_data_size" "128")
ALGORITHM_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=False"
    "algorithm.trajectory_grpo.scheduler=row"
    "algorithm.trajectory_grpo.reducer=token_mean"
    "algorithm.trajectory_grpo.advantage=step_row"
    "algorithm.trajectory_grpo.penalty=step_local"
    "algorithm.trajectory_grpo.filter=off"
    "+algorithm.opsd.v0_prior=0.5"
    "+algorithm.opsd.belief_mult=true"
    "+algorithm.opsd.mult_lambda=0.5"
    "+algorithm.opsd.granularity=token"
    "+algorithm.opsd.signed=true"
    "+algorithm.opsd.skills_dir=skills/webshop"
    "+algorithm.opsd.skill_all=false"
)
DATA_ARGS=(
    "data.train_files=${TRAIN_FILE}" "data.val_files=${VAL_FILE}" "data.seed=0"
    "data.train_batch_size=16" "data.val_batch_size=128"
    "data.max_prompt_length=4096" "data.max_response_length=512"
    "data.filter_overlong_prompts=True" "data.truncation=error" "data.return_raw_chat=True"
)
MODEL_ARGS=(
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
)
ACTOR_ARGS=(
    "actor_rollout_ref.actor.optim.lr=1e-6" "actor_rollout_ref.actor.strategy=fsdp"
    "actor_rollout_ref.actor.ppo_epochs=1" "actor_rollout_ref.actor.ppo_mini_batch_size=64"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.actor.use_dynamic_bsz=False" "actor_rollout_ref.actor.shuffle=False"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
    "actor_rollout_ref.actor.use_kl_loss=True" "actor_rollout_ref.actor.kl_loss_coef=0.01"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.use_invalid_action_penalty=True"
    "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
)
ROLLOUT_ARGS=(
    "actor_rollout_ref.rollout.name=vllm" "++actor_rollout_ref.rollout.seed=0"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.free_cache_engine=False"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.4"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
)
REFERENCE_ARGS=(
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
)
REWARD_ARGS=("reward_model.enable=False" "reward_model.reward_manager=episode")
ENV_ARGS=(
    "env.env_name=Webshop" "env.fairness=true" "env.seed=0"
    "env.history_length=2" "env.max_steps=15" "env.rollout.n=8"
    "env.resources_per_worker.num_cpus=0.1"
    "env.validation_concurrency=${VALIDATION_CONCURRENCY}"
    "++env.webshop.observation_mode=text"
)
TRAINER_ARGS=(
    "trainer.critic_warmup=0" "trainer.logger=['console','swanlab']"
    "trainer.project_name=iclr27_webshop"
    "trainer.experiment_name=agentopsd_qwen2.5_3b"
    "trainer.n_gpus_per_node=4" "trainer.nnodes=1" "trainer.resume_mode=auto"
    "trainer.max_actor_ckpt_to_keep=2" "trainer.max_critic_ckpt_to_keep=2"
    "trainer.save_freq=10" "trainer.test_freq=5"
    "trainer.total_epochs=150" "trainer.total_training_steps=150"
    "trainer.val_before_train=true"
)

if [[ "${LAUNCHER_DRY_RUN:-false}" == true ]]; then
    printf 'module=%s\n' "${TRAINER_MODULE}"
    printf '%s\n' "${ALGORITHM_ARGS[@]}" "${DATA_ARGS[@]}" "${MODEL_ARGS[@]}" "${ACTOR_ARGS[@]}"
    printf '%s\n' "${ROLLOUT_ARGS[@]}" "${REFERENCE_ARGS[@]}" "${REWARD_ARGS[@]}" "${ENV_ARGS[@]}" "${TRAINER_ARGS[@]}"
    if (( $# > 0 )); then printf '%s\n' "$@"; fi
    exit 0
fi

bash "${REPO_ROOT}/scripts/bootstrap_webshop_data.sh"

"${PYTHON_BIN}" -m examples.data_preprocess.prepare "${DATA_PREP_ARGS[@]}"
"${PYTHON_BIN}" -m "${TRAINER_MODULE}" \
    "${ALGORITHM_ARGS[@]}" "${DATA_ARGS[@]}" "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${REFERENCE_ARGS[@]}" \
    "${REWARD_ARGS[@]}" "${ENV_ARGS[@]}" "${TRAINER_ARGS[@]}" "$@"

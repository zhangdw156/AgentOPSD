import os
import subprocess
from itertools import product
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
MODEL_SIZES = ("1.5b", "3b", "7b")
BENCHMARKS = ("alfworld", "webshop")


def _expected_launchers():
    return {
        EXAMPLES_ROOT
        / f"agentopsd_trainer_{model_size}"
        / f"run_{benchmark}.sh"
        for model_size, benchmark in product(MODEL_SIZES, BENCHMARKS)
    }


def _dry_run(path, *extra_args):
    completed = subprocess.run(
        [str(path), *extra_args],
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, "LAUNCHER_DRY_RUN": "true"},
        stdout=subprocess.PIPE,
        text=True,
    )
    lines = completed.stdout.splitlines()
    return lines[0].removeprefix("module="), lines[1:]


def _effective(arguments):
    effective = {}
    for argument in arguments:
        key, separator, value = argument.partition("=")
        if separator:
            effective[key.lstrip("+")] = value
    return effective


def test_examples_contains_exactly_six_agentopsd_launchers():
    expected = _expected_launchers()
    assert len(expected) == 6
    assert set(EXAMPLES_ROOT.rglob("*.sh")) == expected
    assert {
        path.name
        for path in EXAMPLES_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == {
        "data_preprocess",
        "agentopsd_trainer_1.5b",
        "agentopsd_trainer_3b",
        "agentopsd_trainer_7b",
    }
    assert {
        path.name
        for path in (EXAMPLES_ROOT / "data_preprocess").iterdir()
        if path.is_file()
    } == {"__init__.py", "prepare.py"}


def test_launchers_are_standalone_valid_and_user_args_win():
    for launcher in sorted(_expected_launchers()):
        text = launcher.read_text(encoding="utf-8")
        assert os.access(launcher, os.X_OK)
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        assert "source " not in text
        assert "conda" not in text
        assert "mamba" not in text
        assert "export VERL_AGENT_FAIRNESS_CACHE=" not in text
        assert text.count(
            '"${PYTHON_BIN}" -m examples.data_preprocess.prepare'
        ) == 1
        assert text.count('"${PYTHON_BIN}" -m "${TRAINER_MODULE}"') == 1
        assert text.rstrip().endswith('"$@"')

        module, arguments = _dry_run(
            launcher,
            "trainer.total_training_steps=12",
        )
        effective = _effective(arguments)
        assert module == "verl.trainer.main_opsd"
        assert effective["trainer.total_training_steps"] == "12"
        assert effective["algorithm.opsd.granularity"] == (
            "turn" if "alfworld" in launcher.name else "token"
        )
        assert effective["algorithm.opsd.belief_mult"] == "true"
        assert effective["algorithm.opsd.signed"] == "true"
        assert effective["env.fairness"] == "true"


def test_launchers_match_the_fair_comparison_contract():
    for launcher in sorted(_expected_launchers()):
        size = launcher.parent.name.removeprefix("agentopsd_trainer_")
        benchmark = launcher.stem.removeprefix("run_")
        _, arguments = _dry_run(launcher)
        effective = _effective(arguments)

        assert effective["actor_rollout_ref.model.path"].endswith(
            f"/Qwen2.5-{size.upper()}-Instruct"
        )
        assert effective["data.seed"] == "0"
        assert effective["data.train_batch_size"] == "16"
        assert effective["data.val_batch_size"] == "128"
        assert effective["data.max_response_length"] == "512"
        assert effective["data.filter_overlong_prompts"] == "True"
        assert effective["data.truncation"] == "error"
        assert effective["data.return_raw_chat"] == "True"
        assert effective["actor_rollout_ref.actor.optim.lr"] == "1e-6"
        assert effective["actor_rollout_ref.actor.strategy"] == "fsdp"
        assert effective["actor_rollout_ref.actor.ppo_epochs"] == "1"
        assert effective["actor_rollout_ref.actor.use_dynamic_bsz"] == "False"
        assert effective["actor_rollout_ref.actor.shuffle"] == "False"
        assert effective[
            "actor_rollout_ref.actor.ulysses_sequence_parallel_size"
        ] == "1"
        assert effective["actor_rollout_ref.actor.loss_agg_mode"] == "token-mean"
        assert effective["actor_rollout_ref.actor.policy_loss.loss_mode"] == "vanilla"
        assert effective["actor_rollout_ref.actor.use_kl_loss"] == "True"
        assert effective["actor_rollout_ref.actor.kl_loss_coef"] == "0.01"
        assert effective["actor_rollout_ref.actor.kl_loss_type"] == "low_var_kl"
        assert effective[
            "actor_rollout_ref.actor.use_invalid_action_penalty"
        ] == "True"
        assert effective[
            "actor_rollout_ref.actor.invalid_action_penalty_coef"
        ] == "0.1"
        assert effective[
            "actor_rollout_ref.actor.fsdp_config.param_offload"
        ] == "False"
        assert effective["actor_rollout_ref.rollout.seed"] == "0"
        assert effective["actor_rollout_ref.rollout.name"] == "vllm"
        assert effective["actor_rollout_ref.rollout.tensor_model_parallel_size"] == {
            "1.5b": "1",
            "3b": "2",
            "7b": "4",
        }[size]
        assert effective[
            "actor_rollout_ref.rollout.enable_chunked_prefill"
        ] == "False"
        assert effective["actor_rollout_ref.rollout.enforce_eager"] == "False"
        assert effective[
            "actor_rollout_ref.rollout.free_cache_engine"
        ] == "False"
        assert effective[
            "actor_rollout_ref.rollout.val_kwargs.temperature"
        ] == "0.4"
        assert effective[
            "actor_rollout_ref.rollout.val_kwargs.do_sample"
        ] == "True"
        assert effective["actor_rollout_ref.rollout.val_kwargs.n"] == "1"
        assert effective[
            "actor_rollout_ref.ref.fsdp_config.param_offload"
        ] == "True"
        assert effective["reward_model.enable"] == "False"
        assert effective["reward_model.reward_manager"] == "episode"
        assert effective["algorithm.use_kl_in_reward"] == "False"
        assert effective["env.seed"] == "0"
        assert effective["env.history_length"] == "2"
        assert effective["env.rollout.n"] == "8"
        assert effective["env.resources_per_worker.num_cpus"] == "0.1"
        assert effective["trainer.critic_warmup"] == "0"
        assert effective["trainer.n_gpus_per_node"] == "4"
        assert effective["trainer.nnodes"] == "1"
        assert effective["trainer.resume_mode"] == "auto"
        assert effective["trainer.save_freq"] == "10"
        assert effective["trainer.test_freq"] == "5"
        assert effective["trainer.total_epochs"] == "150"
        assert effective["trainer.total_training_steps"] == "150"
        assert effective["trainer.max_actor_ckpt_to_keep"] == "2"
        assert effective["trainer.max_critic_ckpt_to_keep"] == "2"
        assert effective["trainer.logger"] == "['console','swanlab']"
        assert effective["trainer.val_before_train"] == "true"
        assert effective["env.max_steps"] == {
            "alfworld": "50",
            "webshop": "15",
        }[benchmark]
        assert effective["data.max_prompt_length"] == {
            "alfworld": "2048",
            "webshop": "4096",
        }[benchmark]
        assert effective[
            "actor_rollout_ref.actor.ppo_mini_batch_size"
        ] == {
            "alfworld": "256",
            "webshop": "64",
        }[benchmark]

        memory_contract = {
            ("1.5b", "alfworld"): ("64", "64", "64", "0.6", "False"),
            ("3b", "alfworld"): ("32", "32", "32", "0.6", "False"),
            ("7b", "alfworld"): ("8", "8", "8", "0.45", "True"),
            ("1.5b", "webshop"): ("16", "32", "32", "0.6", "False"),
            ("3b", "webshop"): ("8", "16", "16", "0.6", "False"),
            ("7b", "webshop"): ("8", "8", "8", "0.45", "True"),
        }[(size, benchmark)]
        assert effective[
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
        ] == memory_contract[0]
        assert effective[
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
        ] == memory_contract[1]
        assert effective[
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"
        ] == memory_contract[2]
        assert effective[
            "actor_rollout_ref.rollout.gpu_memory_utilization"
        ] == memory_contract[3]
        assert effective[
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload"
        ] == memory_contract[4]


def test_runtime_contracts_are_environment_specific():
    for size in MODEL_SIZES:
        alfworld = (
            EXAMPLES_ROOT
            / f"agentopsd_trainer_{size}"
            / "run_alfworld.sh"
        ).read_text(encoding="utf-8")
        webshop = (
            EXAMPLES_ROOT
            / f"agentopsd_trainer_{size}"
            / "run_webshop.sh"
        ).read_text(encoding="utf-8")

        assert (
            'PYTHON_BIN="${PYTHON_BIN:-/data/zhangdw12/work/verl-agent/'
            '.uv-venv/verl-agent/bin/python3}"'
        ) in alfworld
        assert 'ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"' in alfworld
        assert 'GLIBC_SHIM="${GLIBC_SHIM:-${VERL_AGENT_RUNTIME_ROOT}/lib/' in alfworld
        assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in webshop
        assert "VERL_AGENT_RUNTIME_ROOT" not in webshop
        assert "GLIBC_SHIM" not in webshop


def test_all_six_launchers_compose_real_hydra_configs():
    config_dir = REPO_ROOT / "verl/trainer/config"
    for launcher in sorted(_expected_launchers()):
        module, arguments = _dry_run(launcher)
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_dir),
        ):
            config = compose(
                config_name="ppo_trainer",
                overrides=arguments,
            )

        assert module == "verl.trainer.main_opsd"
        assert config.data.seed == 0
        assert config.algorithm.trajectory_grpo.scheduler == "row"
        assert config.algorithm.trajectory_grpo.reducer == "token_mean"
        assert config.algorithm.trajectory_grpo.advantage == "step_row"
        assert config.algorithm.trajectory_grpo.penalty == "step_local"
        assert config.algorithm.trajectory_grpo.filter == "off"
        assert config.algorithm.opsd.granularity == (
            "turn" if "alfworld" in launcher.name else "token"
        )

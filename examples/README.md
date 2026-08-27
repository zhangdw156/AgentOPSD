# AgentOPSD ICLR Fairness Launchers

The paper-facing experiment surface contains exactly six standalone launchers:

```text
examples/agentopsd_trainer_{1.5b,3b,7b}/run_alfworld.sh
examples/agentopsd_trainer_{1.5b,3b,7b}/run_webshop.sh
```

Every launcher prepares the placeholder parquet files and directly invokes
`verl.trainer.main_opsd`. User-supplied Hydra arguments are appended last and
therefore take precedence. Set `LAUNCHER_DRY_RUN=true` to print the resolved
module and arguments without preparing data or starting Ray.

All six runs use the matching `verl-agent` GRPO fairness contract: seed 0,
train batch 16, group size 8, four GPUs, model-size TP 1/2/4, learning rate
`1e-6`, 150 training steps, validation every 5 steps, checkpointing every 10
steps with two checkpoints retained, SwanLab logging, and `env.fairness=true`.
AgentOPSD-specific settings remain official: ALFWorld uses recursive
turn-level credit (`granularity=turn`), while WebShop uses the official
token-level credit (`granularity=token`). Both use `v0_prior=0.5`,
`belief_mult=true`, `mult_lambda=0.5`, `signed=true`, task-specific
`skills_dir`, and `skill_all=false`.

Fairness manifests are downloaded on first use to the default
`$HOME/.cache/verl-agent/fairness` cache. ALFWorld raw data defaults to
`$HOME/.cache/alfworld`. WebShop raw data and search indexes remain repo-local under
`agent_system/environments/env_package/webshop/webshop/{data,search_engine/indexes}`.

ALFWorld launchers default to
`/data/zhangdw12/work/verl-agent/.uv-venv/verl-agent/bin/python3` and
conditionally preload the GLIBC shim from the same runtime root. WebShop
launchers use `python3` and do not activate conda or mamba; activate the desired
environment before running them.

Fairness validation is exhaustive and chunked at no more than 128 concurrent
environments: ALFWorld evaluates 140 seen plus 134 unseen tasks, while WebShop
evaluates all 500 canonical evaluation tasks.

## Rollout performance

ALFWorld and WebShop now compact each generation/step batch to unfinished
trajectories while scattering observations, histories, task metadata, and
original row indices back to their stable environment slots. ALFWorld also
keeps supported FSDP-vLLM rollout sessions resident for the bounded episode
loop, so model weights and KV-cache allocation are not repeatedly
woken/synchronized between turns. Unsupported rollout backends retain the
legacy per-call context path.

These optimizations do not change AgentOPSD semantics: ALFWorld still uses
turn-level recursive credit and WebShop still uses token-level credit;
SkillBank retrieval, teacher log-prob inputs, persistent WebShop RNG,
fairness task identity, trajectory-GRPO grouping, and per-trajectory
`turn_step` ordering remain attached to the original trajectory rows.

Examples:

```bash
bash examples/agentopsd_trainer_1.5b/run_alfworld.sh
bash examples/agentopsd_trainer_7b/run_webshop.sh
LAUNCHER_DRY_RUN=true \
  bash examples/agentopsd_trainer_3b/run_alfworld.sh \
  trainer.total_training_steps=12
```

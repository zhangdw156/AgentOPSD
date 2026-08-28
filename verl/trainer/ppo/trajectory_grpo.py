# Copyright 2026 The verl-agent team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native trajectory-GRPO configuration contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

NATIVE_TRAJECTORY_GRPO_CONFIG = {
    "scheduler": "row",
    "reducer": "token_mean",
    "advantage": "step_row",
    "penalty": "step_local",
    "filter": "off",
}


def resolve_trajectory_grpo_config(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve defaults and reject every non-native trajectory-GRPO option."""
    if "algorithm" in config:
        algorithm = config["algorithm"]
        if not isinstance(algorithm, Mapping):
            raise ValueError("algorithm config must be a mapping")
        config = algorithm.get("trajectory_grpo", {})
    elif "trajectory_grpo" in config:
        config = config["trajectory_grpo"]

    if not isinstance(config, Mapping):
        raise ValueError("trajectory_grpo config must be a mapping")

    unknown = set(config) - set(NATIVE_TRAJECTORY_GRPO_CONFIG)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported trajectory_grpo fields: {names}")

    resolved = dict(NATIVE_TRAJECTORY_GRPO_CONFIG)
    resolved.update(config)
    for name, expected in NATIVE_TRAJECTORY_GRPO_CONFIG.items():
        actual = resolved[name]
        if actual != expected:
            raise ValueError(
                f"trajectory_grpo.{name}={actual!r} requires native "
                f"value {expected!r}"
            )
    return resolved


def validate_trajectory_grpo_config(
    config: Mapping[str, Any],
) -> None:
    """Validate that trajectory-GRPO uses the only supported native contract."""
    resolve_trajectory_grpo_config(config)


__all__ = [
    "NATIVE_TRAJECTORY_GRPO_CONFIG",
    "resolve_trajectory_grpo_config",
    "validate_trajectory_grpo_config",
]

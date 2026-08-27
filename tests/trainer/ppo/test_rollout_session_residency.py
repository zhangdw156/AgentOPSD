# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

import verl.workers.fsdp_workers as fsdp_workers
from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
)


class _FakeData:
    def __init__(self):
        self.meta_info = {}
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class _FakeShardingManager:
    supports_rollout_session = True

    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.preprocess_count = 0
        self.postprocess_count = 0
        self.enter_error = None
        self.exit_error = None
        self._tainted = False

    def __enter__(self):
        self.enter_count += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_count += 1
        if self.exit_error is not None:
            self._tainted = True
            raise self.exit_error

    def preprocess_data(self, data):
        self.preprocess_count += 1
        return data

    def postprocess_data(self, data):
        self.postprocess_count += 1
        return data


class _FakeRollout:
    def __init__(self):
        self.generate_count = 0

    def generate_sequences(self, prompts):
        self.generate_count += 1
        return prompts


class _FakeDevice:
    def __init__(self):
        self.empty_cache_count = 0

    def current_device(self):
        return "fake-device"

    def empty_cache(self):
        self.empty_cache_count += 1

    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return 0

    def mem_get_info(self):
        return 1, 1


@pytest.fixture
def worker_and_device(monkeypatch):
    device = _FakeDevice()
    monkeypatch.setattr(
        fsdp_workers,
        "get_torch_device",
        lambda: device,
    )
    monkeypatch.setattr(
        fsdp_workers,
        "log_gpu_memory_usage",
        lambda *args, **kwargs: None,
    )

    worker = object.__new__(ActorRolloutRefWorker)
    worker._is_rollout = True
    worker._is_actor = True
    worker._rollout_session_entering = False
    worker._rollout_session_active = False
    worker._rollout_session_tainted = False
    worker.rollout_sharding_manager = _FakeShardingManager()
    worker.rollout = _FakeRollout()
    worker.generation_config = SimpleNamespace(
        eos_token_id=2,
        pad_token_id=0,
    )
    worker.tokenizer = SimpleNamespace(
        eos_token_id=3,
        pad_token_id=1,
    )
    return worker, device


def test_worker_session_reuses_resident_weights_and_cache(
    worker_and_device,
):
    worker, device = worker_and_device

    worker.begin_rollout_session()
    first = worker.generate_sequences(_FakeData())
    second = worker.generate_sequences(_FakeData())
    worker.end_rollout_session()

    assert worker.rollout_sharding_manager.enter_count == 1
    assert worker.rollout_sharding_manager.exit_count == 1
    assert worker.rollout_sharding_manager.preprocess_count == 2
    assert worker.rollout_sharding_manager.postprocess_count == 2
    assert worker.rollout.generate_count == 2
    assert first.to_calls == ["fake-device", "cpu"]
    assert second.to_calls == ["fake-device", "cpu"]
    assert device.empty_cache_count == 1


def test_worker_session_guards_taint_update_and_registration(
    worker_and_device,
):
    worker, _ = worker_and_device

    begin_attrs = getattr(
        ActorRolloutRefWorker.begin_rollout_session,
        MAGIC_ATTR,
    )
    end_attrs = getattr(
        ActorRolloutRefWorker.end_rollout_session,
        MAGIC_ATTR,
    )
    assert begin_attrs["dispatch_mode"] == Dispatch.ONE_TO_ALL
    assert end_attrs["dispatch_mode"] == Dispatch.ONE_TO_ALL

    worker.begin_rollout_session()
    with pytest.raises(
        RuntimeError,
        match="entering, active, or tainted",
    ):
        worker.update_actor(_FakeData())
    worker.rollout_sharding_manager.exit_error = ValueError(
        "exit failed"
    )
    with pytest.raises(ValueError, match="exit failed"):
        worker.end_rollout_session()
    assert worker._rollout_session_tainted
    with pytest.raises(RuntimeError, match="tainted"):
        worker.begin_rollout_session()
    with pytest.raises(
        RuntimeError,
        match="entering, active, or tainted",
    ):
        worker.update_actor(_FakeData())


def test_unsupported_and_async_workers_use_safe_fallback(
    worker_and_device,
):
    worker, device = worker_and_device
    worker.rollout_sharding_manager.supports_rollout_session = False

    worker.begin_rollout_session()
    worker.generate_sequences(_FakeData())
    worker.generate_sequences(_FakeData())
    worker.end_rollout_session()

    assert worker.rollout_sharding_manager.enter_count == 2
    assert worker.rollout_sharding_manager.exit_count == 2
    assert device.empty_cache_count == 2

    async_worker = object.__new__(AsyncActorRolloutRefWorker)
    with pytest.raises(NotImplementedError):
        async_worker.begin_rollout_session()
    with pytest.raises(NotImplementedError):
        async_worker.end_rollout_session()


@pytest.fixture
def fsdp_vllm_module(monkeypatch):
    third_party_vllm = ModuleType("verl.third_party.vllm")
    third_party_vllm.LLM = object
    third_party_vllm.vllm_version = "0.8.0"
    third_party_vllm.parallel_state = SimpleNamespace()

    vllm_utils = ModuleType("verl.utils.vllm_utils")
    vllm_utils.TensorLoRARequest = object
    vllm_utils.VLLMHijack = SimpleNamespace(
        hijack=lambda: None
    )
    vllm_utils.is_version_ge = lambda **kwargs: False
    vllm_utils.patch_vllm_moe_model_weight_loader = (
        lambda model: None
    )

    monkeypatch.setitem(
        sys.modules,
        "verl.third_party.vllm",
        third_party_vllm,
    )
    monkeypatch.setitem(
        sys.modules,
        "verl.utils.vllm_utils",
        vllm_utils,
    )
    sys.modules.pop(
        "verl.workers.sharding_manager.fsdp_vllm",
        None,
    )
    module = importlib.import_module(
        "verl.workers.sharding_manager.fsdp_vllm"
    )
    monkeypatch.setattr(
        module,
        "log_gpu_memory_usage",
        lambda *args, **kwargs: None,
    )
    return module


class _FakeRNGDevice:
    def __init__(self):
        self.state = "train"
        self.set_calls = []
        self.empty_cache_count = 0
        self.fail_restore = False

    def get_rng_state(self):
        return self.state

    def set_rng_state(self, state):
        if state == "train" and self.fail_restore:
            raise RuntimeError("RNG restore failed")
        self.state = state
        self.set_calls.append(state)

    def empty_cache(self):
        self.empty_cache_count += 1

    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return 0

    def mem_get_info(self):
        return 1, 1


class _FakeFSDPModule:
    def __init__(self):
        self._fsdp_wrapped_module = object()
        self.train_count = 0

    def state_dict(self):
        return {"weight": object()}

    def train(self):
        self.train_count += 1


class _FakeVLLMEngine:
    def __init__(self):
        self.awake = False
        self.wake_count = 0
        self.sleep_count = 0

    def wake_up(self, tags=None):
        if not self.awake:
            self.wake_count += 1
        self.awake = True

    def sleep(self, level):
        assert level == 1
        self.sleep_count += 1
        self.awake = False


def _make_manager(module, device, monkeypatch):
    manager = object.__new__(
        module.FSDPVLLMShardingManager
    )
    manager.module = _FakeFSDPModule()
    manager.inference_engine = _FakeVLLMEngine()
    manager.device_mesh = object()
    manager.offload_param = False
    manager.full_params = False
    manager.layered_summon = False
    manager.base_sync_done = True
    manager.gen_random_states = "generation"
    manager.torch_random_states = "train"
    manager._enter_actor_params_loaded = False
    manager._enter_rollout_awake = False
    manager._enter_rng_switch_attempted = False
    manager._enter_rng_switched = False
    manager._enter_train_mode_restore_needed = False
    manager._enter_cache_cleanup_needed = False
    manager._enter_succeeded = False
    manager._tainted = False
    manager._taint_reason = None
    manager.update_params = (
        lambda params, peft_config=None: None
    )
    monkeypatch.setattr(
        module,
        "get_torch_device",
        lambda: device,
    )
    performance = importlib.import_module(
        "verl.utils.debug.performance"
    )
    monkeypatch.setattr(
        performance,
        "get_torch_device",
        lambda: device,
    )
    return manager


def test_fsdp_vllm_session_persists_rng_and_weights(
    fsdp_vllm_module,
    monkeypatch,
):
    assert (
        fsdp_vllm_module.FSDPVLLMShardingManager
        .supports_rollout_session
    )
    device = _FakeRNGDevice()
    manager = _make_manager(
        fsdp_vllm_module,
        device,
        monkeypatch,
    )

    manager.__enter__()
    assert device.state == "generation"
    assert manager.inference_engine.awake
    assert manager._enter_succeeded

    device.state = "generation-next"
    manager.__exit__(None, None, None)
    assert device.state == "train"
    assert manager.gen_random_states == "generation-next"
    assert manager.inference_engine.wake_count == 1
    assert manager.inference_engine.sleep_count == 1
    assert manager.module.train_count == 1
    assert not manager._enter_succeeded
    assert not manager._tainted


def test_fsdp_vllm_cleanup_failure_taints_and_blocks_reuse(
    fsdp_vllm_module,
    monkeypatch,
):
    device = _FakeRNGDevice()
    manager = _make_manager(
        fsdp_vllm_module,
        device,
        monkeypatch,
    )
    manager.__enter__()
    device.fail_restore = True

    with pytest.raises(RuntimeError, match="RNG restore failed"):
        manager.__exit__(None, None, None)

    assert manager._tainted
    assert manager._enter_rng_switched
    with pytest.raises(RuntimeError, match="tainted"):
        manager.__enter__()

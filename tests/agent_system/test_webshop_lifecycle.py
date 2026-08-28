# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import agent_system.environments as environments_package
import agent_system.environments.env_package as env_package
from agent_system.environments import env_manager, fairness
from verl.trainer import main_opsd

REPO_ROOT = Path(__file__).parents[2]


def _load_webshop_text_env_with_stubs(monkeypatch):
    gym_module = types.ModuleType("gym")
    gym_module.Env = object
    monkeypatch.setitem(sys.modules, "gym", gym_module)

    bs4_module = types.ModuleType("bs4")
    bs4_module.BeautifulSoup = object
    bs4_element_module = types.ModuleType("bs4.element")
    bs4_element_module.Comment = object
    monkeypatch.setitem(sys.modules, "bs4", bs4_module)
    monkeypatch.setitem(sys.modules, "bs4.element", bs4_element_module)

    class FakeFlask:
        def __init__(self, _name):
            pass

        def route(self, *_args, **_kwargs):
            return lambda function: function

    flask_module = types.ModuleType("flask")
    flask_module.Flask = FakeFlask
    monkeypatch.setitem(sys.modules, "flask", flask_module)

    engine_module = types.ModuleType("web_agent_site.engine.engine")
    engine_module.load_products = lambda **_kwargs: ([], {}, {}, {})
    engine_module.init_search_engine = lambda **_kwargs: object()
    engine_module.get_top_n_product_from_keywords = lambda *_a, **_k: []
    engine_module.map_action_to_html = lambda *_a, **_k: ("", "")
    engine_module.parse_action = lambda action: (action, "")
    engine_module.get_product_per_page = lambda *_a, **_k: []
    engine_module.ACTION_TO_TEMPLATE = {}
    engine_module.END_BUTTON = "Buy Now"
    engine_module.NEXT_PAGE = "Next >"
    engine_module.PREV_PAGE = "< Prev"
    engine_module.BACK_TO_SEARCH = "Back to Search"

    goal_module = types.ModuleType("web_agent_site.engine.goal")
    goal_module.get_reward = lambda *_a, **_k: 0
    goal_module.get_goals = lambda *_a, **_k: []

    utils_module = types.ModuleType("web_agent_site.utils")
    utils_module.DEFAULT_FILE_PATH = "items.json"
    utils_module.DEFAULT_ATTR_PATH = "attrs.json"
    utils_module.FEAT_CONV = "features.pt"
    utils_module.FEAT_IDS = "ids.pt"
    utils_module.random_idx = lambda _weights: 0

    web_agent_site = types.ModuleType("web_agent_site")
    web_agent_site.__path__ = []
    engine_package = types.ModuleType("web_agent_site.engine")
    engine_package.__path__ = []
    monkeypatch.setitem(sys.modules, "web_agent_site", web_agent_site)
    monkeypatch.setitem(sys.modules, "web_agent_site.engine", engine_package)
    monkeypatch.setitem(
        sys.modules,
        "web_agent_site.engine.engine",
        engine_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "web_agent_site.engine.goal",
        goal_module,
    )
    monkeypatch.setitem(sys.modules, "web_agent_site.utils", utils_module)

    module_path = (
        REPO_ROOT
        / "agent_system/environments/env_package/webshop/webshop/"
        "web_agent_site/envs/web_agent_text_env.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_lifecycle_web_agent_text_env",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_webshop_pool_module(monkeypatch):
    gym_module = types.ModuleType("gym")
    gym_module.Env = object
    monkeypatch.setitem(sys.modules, "gym", gym_module)
    module_path = (
        REPO_ROOT
        / "agent_system/environments/env_package/webshop/envs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_lifecycle_webshop_pool",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lazy_environment_manager_releases_and_recreates():
    events = []
    next_id = 0

    class FakeManager:
        def __init__(self, manager_id):
            self.manager_id = manager_id

        def identity(self):
            return self.manager_id

        def close(self):
            events.append(f"close:{self.manager_id}")

    def factory():
        nonlocal next_id
        next_id += 1
        events.append(f"build:{next_id}")
        return FakeManager(next_id)

    manager = env_manager.LazyEnvironmentManager(factory)
    assert manager.identity() == 1
    manager.release()
    assert manager.identity() == 2
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.identity()
    assert events == ["build:1", "close:1", "build:2", "close:2"]


@pytest.mark.parametrize("val_only", [False, True])
def test_fair_webshop_pool_is_lazy_and_val_only_skips_train(
    monkeypatch,
    val_only,
):
    events = []
    training_rngs = []

    class FakeRawPool:
        def __init__(self, label):
            self.label = label
            events.append(f"build:{label}")

        def close(self):
            events.append(f"close:{self.label}")

    class FakeManager:
        def __init__(self, raw_pool, _projection, _config):
            self.raw_pool = raw_pool

        def identity(self):
            return self.raw_pool.label

        def close(self):
            self.raw_pool.close()

    def build_webshop_envs(*, is_train, rng=None, **_kwargs):
        if is_train:
            training_rngs.append(rng)
        return FakeRawPool("train" if is_train else "validation")

    webshop_module = types.ModuleType(
        "agent_system.environments.env_package.webshop"
    )
    webshop_module.build_webshop_envs = build_webshop_envs
    webshop_module.webshop_projection = lambda actions: actions
    monkeypatch.setattr(
        environments_package,
        "env_package",
        env_package,
        raising=False,
    )
    monkeypatch.setattr(
        env_package,
        "webshop",
        webshop_module,
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "agent_system.environments.env_package.webshop",
        webshop_module,
    )
    monkeypatch.setattr(
        env_manager,
        "WebshopEnvironmentManager",
        FakeManager,
    )
    monkeypatch.setattr(
        fairness,
        "canonical_validation_splits",
        lambda _environment: ("evaluation",),
    )
    monkeypatch.setattr(
        fairness,
        "canonical_validation_chunks",
        lambda _environment, _split, *, concurrency: ((0, 1),),
    )

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "Webshop",
                "seed": 0,
                "fairness": True,
                "rollout": {"n": 8},
                "resources_per_worker": {
                    "num_cpus": 0.1,
                    "num_gpus": 0,
                },
                "webshop": {
                    "use_small": True,
                    "human_goals": False,
                },
            },
            "data": {
                "train_batch_size": 16,
                "val_batch_size": 128,
            },
            "trainer": {"val_only": val_only},
        }
    )

    train_envs, validation_envs = env_manager.make_envs(config)
    assert events == []
    if val_only:
        assert train_envs is None
        assert training_rngs == []
    else:
        assert train_envs.identity() == "train"
        first_rng = training_rngs[0]
        chunk_iterator = validation_envs.iter_chunks()
        chunk = next(chunk_iterator)
        assert chunk.manager.identity() == "validation"
        chunk_iterator.close()
        assert train_envs.identity() == "train"
        assert training_rngs == [first_rng, first_rng]
        train_envs.close()


def test_nonfair_webshop_pools_are_lazy_phase_exclusive_and_reuse_rng(
    monkeypatch,
):
    events = []
    live_pools = set()
    peak_live_pools = 0
    rngs = {"train": [], "validation": []}

    class FakeRawPool:
        def __init__(self, label):
            nonlocal peak_live_pools
            self.label = label
            self.closed = False
            live_pools.add(label)
            peak_live_pools = max(peak_live_pools, len(live_pools))
            events.append(f"build:{label}")

        def close(self):
            if self.closed:
                return
            self.closed = True
            live_pools.remove(self.label)
            events.append(f"close:{self.label}")

    class FakeManager:
        def __init__(self, raw_pool, _projection, _config):
            self.raw_pool = raw_pool

        def identity(self):
            return self.raw_pool.label

        def close(self):
            self.raw_pool.close()

    def build_webshop_envs(*, is_train, rng=None, **_kwargs):
        label = "train" if is_train else "validation"
        rngs[label].append(rng)
        return FakeRawPool(label)

    webshop_module = types.ModuleType(
        "agent_system.environments.env_package.webshop"
    )
    webshop_module.build_webshop_envs = build_webshop_envs
    webshop_module.webshop_projection = lambda actions: actions
    monkeypatch.setattr(
        environments_package,
        "env_package",
        env_package,
        raising=False,
    )
    monkeypatch.setattr(
        env_package,
        "webshop",
        webshop_module,
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "agent_system.environments.env_package.webshop",
        webshop_module,
    )
    monkeypatch.setattr(
        env_manager,
        "WebshopEnvironmentManager",
        FakeManager,
    )

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "Webshop",
                "seed": 0,
                "fairness": False,
                "rollout": {"n": 8},
                "resources_per_worker": {
                    "num_cpus": 0.1,
                    "num_gpus": 0,
                },
                "webshop": {
                    "use_small": True,
                    "human_goals": False,
                },
            },
            "data": {
                "train_batch_size": 16,
                "val_batch_size": 128,
            },
            "trainer": {"val_only": False},
        }
    )

    train_envs, validation_envs = env_manager.make_envs(config)

    assert events == []
    assert train_envs.identity() == "train"
    assert live_pools == {"train"}
    assert validation_envs.identity() == "validation"
    assert live_pools == {"validation"}
    assert train_envs.identity() == "train"
    assert live_pools == {"train"}
    assert validation_envs.identity() == "validation"
    assert live_pools == {"validation"}

    train_envs.close()
    validation_envs.close()

    assert peak_live_pools == 1
    assert len(rngs["train"]) == 2
    assert rngs["train"][0] is rngs["train"][1]
    assert len(rngs["validation"]) == 2
    assert rngs["validation"][0] is rngs["validation"][1]
    assert events == [
        "build:train",
        "close:train",
        "build:validation",
        "close:validation",
        "build:train",
        "close:train",
        "build:validation",
        "close:validation",
    ]


def test_webshop_environment_close_closes_searcher_once(monkeypatch):
    module = _load_webshop_text_env_with_stubs(monkeypatch)
    close_calls = []

    class FakeSearcher:
        def close(self):
            close_calls.append("close")

    server = object.__new__(module.SimServer)
    server.search_engine = FakeSearcher()
    server.user_sessions = {"session": {}}
    environment = object.__new__(module.WebAgentTextEnv)
    environment.server = server
    environment._owns_server = True

    environment.close()
    environment.close()

    assert close_calls == ["close"]
    assert server.search_engine is None
    assert server.user_sessions == {}


def test_webshop_pool_kills_all_workers_when_close_fails(monkeypatch):
    module = _load_webshop_pool_module(monkeypatch)
    workers = [
        SimpleNamespace(
            close=SimpleNamespace(remote=lambda name=name: name),
        )
        for name in ("first", "second", "third")
    ]
    killed = []

    def fake_get(reference):
        if reference == "second":
            raise RuntimeError("close failed")

    monkeypatch.setattr(
        module,
        "ray",
        SimpleNamespace(get=fake_get, kill=killed.append),
    )
    pool = object.__new__(module.WebshopMultiProcessEnv)
    pool._workers = list(workers)
    pool._closed = False

    with pytest.raises(RuntimeError, match="close failed"):
        pool.close()

    assert killed == workers
    assert pool._workers == []
    assert pool._closed is True


def test_webshop_constructor_kills_partially_created_workers(monkeypatch):
    module = _load_webshop_pool_module(monkeypatch)
    created = []
    killed = []

    class FakeActorClass:
        def remote(self, *_args, **_kwargs):
            if len(created) == 2:
                raise RuntimeError("construction failed")
            worker = SimpleNamespace(name=f"worker-{len(created)}")
            created.append(worker)
            return worker

    fake_actor_class = FakeActorClass()
    monkeypatch.setattr(
        module,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: True,
            remote=lambda **_kwargs: (
                lambda _worker_cls: fake_actor_class
            ),
            kill=killed.append,
        ),
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        module.WebshopMultiProcessEnv(
            seed=0,
            env_num=4,
            group_n=1,
            resources_per_worker={"num_cpus": 0.1},
            is_train=True,
            env_kwargs={"fairness": False},
        )

    assert killed == created


def test_main_opsd_finally_closes_environments():
    events = []

    class Environment:
        def close(self):
            events.append("close")

    trainer = SimpleNamespace(
        envs=Environment(),
        val_envs=Environment(),
        init_workers=lambda: events.append("init"),
        fit=lambda: (_ for _ in ()).throw(
            RuntimeError("training failed")
        ),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        main_opsd._fit_trainer_with_cleanup(trainer)

    assert events == ["init", "close", "close"]

from __future__ import annotations

import datasets

from examples.data_preprocess.prepare import (
    build_placeholder_dataset,
    prepare_datasets,
)


def test_text_placeholder_dataset_is_local_and_environment_driven(
    monkeypatch,
):
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network dataset loading is forbidden")
        ),
    )

    dataset = build_placeholder_dataset(
        mode="text",
        size=2,
        split="train",
    )

    assert len(dataset) == 2
    assert dataset.column_names == [
        "data_source",
        "prompt",
        "ability",
        "extra_info",
    ]
    assert dataset[0]["prompt"] == [
        {"role": "user", "content": ""}
    ]


def test_prepare_datasets_writes_requested_parquet_sizes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepare must not fetch Geometry3K")
        ),
    )
    train_path, test_path = prepare_datasets(
        mode="text",
        local_dir=str(tmp_path),
        train_data_size=3,
        val_data_size=2,
    )

    assert train_path.is_file()
    assert test_path.is_file()
    assert len(datasets.Dataset.from_parquet(str(train_path))) == 3
    assert len(datasets.Dataset.from_parquet(str(test_path))) == 2

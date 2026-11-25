import datasets
import torch
from omegaconf import OmegaConf

from verl.utils.dataset.modality_batch_sampler import (
    ModalityGroupedSampler,
    maybe_modality_grouped_sampler,
    video_row_mask,
)


def _mixed_dataset():
    images = datasets.Dataset.from_dict({"prompt": [f"image-{i}" for i in range(10)], "images": ["x"] * 10})
    videos = datasets.Dataset.from_dict(
        {
            "prompt": [f"video-{i}" for i in range(7)],
            "videos": [[{"video": f"v{i}.mp4"}] for i in range(7)],
        }
    )
    frame = datasets.concatenate_datasets([images, videos])

    class _Wrap:
        dataframe = frame

    return _Wrap()


def test_video_mask_ignores_null_image_rows():
    dataset = _mixed_dataset()
    mask = video_row_mask(dataset.dataframe)
    assert mask.tolist() == [False] * 10 + [True] * 7


def test_each_batch_is_a_single_modality():
    dataset = _mixed_dataset()
    generator = torch.Generator().manual_seed(0)
    sampler = ModalityGroupedSampler(dataset, batch_size=4, shuffle=True, generator=generator)
    order = list(iter(sampler))
    assert len(order) == 12
    assert len(order) == len(sampler)
    for start in range(0, len(order), 4):
        batch = order[start : start + 4]
        kinds = {index >= 10 for index in batch}
        assert kinds == {False} or kinds == {True}
    assert any(index < 10 for index in order)
    assert any(index >= 10 for index in order)


def test_resume_replays_the_same_order():
    dataset = _mixed_dataset()
    generator = torch.Generator().manual_seed(1)
    sampler = ModalityGroupedSampler(dataset, batch_size=4, shuffle=True, generator=generator)
    iterator = iter(sampler)
    seen = [next(iterator) for _ in range(6)]
    state = iterator.state_dict()
    restored = iter(sampler)
    restored.load_state_dict(state)
    assert seen + list(restored) == list(iter(ModalityGroupedSampler(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(1))))


def test_image_only_keeps_the_ordinary_sampler():
    frame = datasets.Dataset.from_dict({"prompt": ["a", "b"], "images": ["x", "y"]})

    class _Wrap:
        dataframe = frame

    config = OmegaConf.create({"shuffle": True, "seed": 0, "train_batch_size": 2, "video_key": "videos"})
    assert maybe_modality_grouped_sampler(config, _Wrap(), torch.Generator().manual_seed(0)) is None

"""Yield training indices so one batch is either all images or all videos.

Image-only and video-only datasets keep the ordinary sampler. A mixed dataset
is split by whether the ``videos`` column is non-empty. Each side is shuffled
on its own, cut into full batches, and those batches are then shuffled together.
A remainder smaller than the batch size is dropped, matching ``drop_last``.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch.utils.data import Sampler
from torchdata.stateful_dataloader.stateful import Stateful


def video_row_mask(dataframe, video_key: str = "videos") -> np.ndarray:
    """True when a row carries at least one video. Does not read image bytes."""
    n = len(dataframe)
    if video_key not in dataframe.column_names:
        return np.zeros(n, dtype=bool)
    column = dataframe.data.column(video_key)
    if pa.types.is_null(column.type):
        return np.zeros(n, dtype=bool)
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        lengths = pc.fill_null(pc.list_value_length(column), 0)
        flags = pc.greater(lengths, 0).fill_null(False)
        return np.array(flags.to_pylist(), dtype=bool)
    return np.array(pc.is_valid(column).fill_null(False).to_pylist(), dtype=bool)


def _complete_count(n_rows: int, batch_size: int) -> int:
    return (int(n_rows) // int(batch_size)) * int(batch_size)


class ModalityGroupedSampler(Sampler[int]):
    """Sampler whose consecutive ``batch_size`` indices share one modality."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        video_key: str = "videos",
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.generator = generator if generator is not None else torch.Generator()
        frame = dataset.dataframe
        mask = video_row_mask(frame, video_key)
        indices = np.arange(len(frame), dtype=np.int64)
        self.image_indices = indices[~mask].tolist()
        self.video_indices = indices[mask].tolist()
        self.num_samples = _complete_count(len(self.image_indices), self.batch_size) + _complete_count(
            len(self.video_indices), self.batch_size
        )
        if self.num_samples < self.batch_size:
            raise ValueError(
                "Modality grouping produced no full batch: "
                f"images={len(self.image_indices)} videos={len(self.video_indices)} batch_size={self.batch_size}"
            )
        dropped_images = len(self.image_indices) % self.batch_size
        dropped_videos = len(self.video_indices) % self.batch_size
        print(
            "Modality-grouped batches: "
            f"images={len(self.image_indices)} (drop {dropped_images}), "
            f"videos={len(self.video_indices)} (drop {dropped_videos}), "
            f"batch_size={self.batch_size}, steps_per_epoch={self.num_samples // self.batch_size}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        return _ModalityGroupedIterator(self)

    def build_order(self) -> list[int]:
        image_batches = self._batches(self.image_indices)
        video_batches = self._batches(self.video_indices)
        batches = image_batches + video_batches
        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[i] for i in order]
        return [index for batch in batches for index in batch]

    def _batches(self, indices: list[int]) -> list[list[int]]:
        if self.shuffle and len(indices) > 1:
            perm = torch.randperm(len(indices), generator=self.generator).tolist()
            indices = [indices[i] for i in perm]
        usable = _complete_count(len(indices), self.batch_size)
        indices = indices[:usable]
        return [indices[start : start + self.batch_size] for start in range(0, usable, self.batch_size)]


class _ModalityGroupedIterator(Iterator[int], Stateful):
    _GENERATOR = "generator"
    _YIELDED = "yielded"

    def __init__(self, sampler: ModalityGroupedSampler) -> None:
        self.sampler = sampler
        self.generator_state = sampler.generator.get_state()
        self.order = sampler.build_order()
        self.yielded = 0
        self.next_yielded = None

    def __next__(self) -> int:
        if self.next_yielded is not None:
            target = self.next_yielded
            self.next_yielded = None
            while self.yielded < target:
                next(self)
        if self.yielded >= len(self.order):
            raise StopIteration
        index = self.order[self.yielded]
        self.yielded += 1
        return index

    def state_dict(self) -> dict:
        return {self._GENERATOR: self.generator_state, self._YIELDED: self.yielded}

    def load_state_dict(self, state_dict: dict) -> None:
        self.generator_state = state_dict[self._GENERATOR]
        self.sampler.generator.set_state(self.generator_state)
        self.order = self.sampler.build_order()
        self.yielded = 0
        self.next_yielded = int(state_dict[self._YIELDED])


def maybe_modality_grouped_sampler(data_config, dataset, generator: torch.Generator):
    """Return a grouped sampler when the dataset contains both images and videos."""
    frame = getattr(dataset, "dataframe", None)
    if frame is None:
        return None
    video_key = data_config.get("video_key", "videos")
    mask = video_row_mask(frame, video_key)
    if not bool(mask.any()) or bool(mask.all()):
        return None
    batch_size = data_config.get("gen_batch_size", None)
    if batch_size is None:
        batch_size = data_config.train_batch_size
    return ModalityGroupedSampler(
        dataset,
        int(batch_size),
        video_key=video_key,
        shuffle=bool(data_config.get("shuffle", False)),
        generator=generator,
    )

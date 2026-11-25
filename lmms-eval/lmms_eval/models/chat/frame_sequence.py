"""Pack a run of timestamped images into one video.

Some tasks sample a video themselves and hand each frame to the chat template
as its own image, with a timestamp label in between. Qwen then treats those
frames as unrelated images. A run of at least two such pairs is the same
video: this rewrites that run into one ``video`` element so the processor
emits ``video_token_id`` and a temporal grid. Pruners that already run on
video tokens then see it. A lone image, or an image that is not preceded by
a timestamp label, stays an image.
"""

from __future__ import annotations

import re

from loguru import logger as eval_logger

_TIMESTAMP_TEXT = re.compile(
    r"^\s*(?:timestamp\s*:\s*)?<?\s*(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\s*>?\s*\.?\s*$",
    re.IGNORECASE,
)
_PIXEL_KEYS = ("min_pixels", "max_pixels", "total_pixels")


def _timestamp_seconds(text: str) -> float | None:
    if not isinstance(text, str) or len(text) > 80:
        return None
    match = _TIMESTAMP_TEXT.match(text.strip())
    if match is None:
        return None
    return float(match.group(1))


def _is_video(item: dict) -> bool:
    return item.get("type") == "video" or "video" in item


def _image_source(item: dict):
    if item.get("type") not in ("image", "image_url"):
        return None
    if "image" in item:
        return item["image"]
    if "image_url" in item:
        return item["image_url"]
    if "url" in item:
        return item["url"]
    return None


def _frame_run(content: list, start: int):
    frames = []
    timestamps = []
    index = start
    while index + 1 < len(content):
        text = content[index]
        image = content[index + 1]
        if not isinstance(text, dict) or not isinstance(image, dict):
            break
        if text.get("type") != "text":
            break
        seconds = _timestamp_seconds(text.get("text", ""))
        source = _image_source(image)
        if seconds is None or source is None:
            break
        frames.append(source)
        timestamps.append(seconds)
        index += 2
    if not frames:
        return None
    return frames, timestamps, index


def pack_timestamped_frames(conversations: list, video_kwargs: dict | None = None) -> list:
    """Rewrite conversations in place.

    Returns one entry per video, in the order ``process_vision_info`` visits
    them. An entry is the frame timestamps in seconds, or ``None`` when that
    video was already a video and its metadata should be left alone.
    """
    video_kwargs = video_kwargs or {}
    stamps: list = []
    packed = 0
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            rewritten = []
            index = 0
            while index < len(content):
                run = _frame_run(content, index)
                if run is not None and len(run[0]) >= 2:
                    frames, timestamps, end = run
                    item = {"type": "video", "video": frames}
                    for key in _PIXEL_KEYS:
                        if key in video_kwargs:
                            item[key] = video_kwargs[key]
                    rewritten.append(item)
                    stamps.append(timestamps)
                    packed += 1
                    eval_logger.debug(
                        f"Packed {len(frames)} timestamped images into one video "
                        f"({timestamps[0]:.3f}s–{timestamps[-1]:.3f}s)"
                    )
                    index = end
                    continue
                item = content[index]
                rewritten.append(item)
                if isinstance(item, dict) and _is_video(item):
                    stamps.append(None)
                index += 1
            message["content"] = rewritten
    if packed:
        eval_logger.info(f"Packed {packed} timestamped image sequence(s) into video input(s)")
    return stamps


def apply_frame_timestamps(videos: list, metadatas: list, stamps: list) -> None:
    """Point processor timestamps at the labels that came with the frames.

    Qwen derives ``<t seconds>`` from ``frames_indices / fps`` and then averages
    each temporal patch. Setting ``fps`` to 1 and ``frames_indices`` to the
    label values keeps those seconds.
    """
    if len(videos) != len(metadatas) or len(videos) != len(stamps):
        raise RuntimeError(
            f"timestamped-frame videos ({len(stamps)}) do not match decoded videos ({len(videos)})"
        )
    for video, metadata, timestamps in zip(videos, metadatas, stamps):
        if not timestamps:
            continue
        count = int(video.shape[0])
        values = list(timestamps)
        if len(values) < count:
            values.extend([values[-1]] * (count - len(values)))
        values = values[:count]
        if isinstance(metadata, dict):
            metadata["fps"] = 1.0
            metadata["frames_indices"] = values
            metadata["total_num_frames"] = count
        else:
            metadata.fps = 1.0
            metadata.frames_indices = values
            metadata.total_num_frames = count

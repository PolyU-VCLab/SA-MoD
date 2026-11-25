import os
import time
from collections import defaultdict
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger as eval_logger

_THROUGHPUT_METRICS_HISTORY: List[Dict[str, Any]] = []
_GEN_METRICS_ENABLED: bool = False

# Keys that are identifiers / already aggregated — do not emit avg_<key>.
_SKIP_AVG_KEYS = frozenset(
    {
        "total_gen_tokens",
        "total_elapsed_time",
        "avg_speed",
        "total_requests",
        "request_count",
        "num_requests",
        "rank",
        "num_steps",
        "ttft",
        "tpot",
        "task",
        "per_sample_kv_cache_mb",
        "kv_cache_mb",
    }
)

_TRUTHY = {"1", "true", "yes", "on", "y"}


def _env_flag_enabled() -> bool:
    raw = os.environ.get("LMMS_LOG_GEN_METRICS", "")
    return str(raw).strip().lower() in _TRUTHY


def set_gen_metrics_enabled(enabled: bool) -> None:
    """Enable/disable generation efficiency metrics for this process."""
    global _GEN_METRICS_ENABLED
    _GEN_METRICS_ENABLED = bool(enabled)
    if _GEN_METRICS_ENABLED:
        eval_logger.info("Generation efficiency metrics enabled (TTFT/throughput/KV cache)")


def is_gen_metrics_enabled() -> bool:
    """True when CLI/env opted into generation efficiency metrics."""
    return _GEN_METRICS_ENABLED or _env_flag_enabled()


def reset_logged_metrics() -> None:
    """Reset in-memory throughput metrics captured during a run."""

    _THROUGHPUT_METRICS_HISTORY.clear()


def get_logged_metrics_history() -> List[Dict[str, Any]]:
    """Return captured throughput metrics in collection order."""

    return list(_THROUGHPUT_METRICS_HISTORY)


def gather_logged_metrics() -> None:
    """Merge per-rank metric histories so rank-0 summarize sees all GPUs.

    Must be called on every rank (collective). No-op when distributed is unused.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        return
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    local = list(_THROUGHPUT_METRICS_HISTORY)
    gathered: List[Optional[List[Dict[str, Any]]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    merged: List[Dict[str, Any]] = []
    for part in gathered:
        if part:
            merged.extend(part)
    _THROUGHPUT_METRICS_HISTORY.clear()
    _THROUGHPUT_METRICS_HISTORY.extend(merged)


def _aggregate_metric_list(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}

    total_gen_tokens = 0.0
    total_elapsed_time = 0.0
    total_requests = 0.0
    avg_speed_vals: List[float] = []
    additional_numeric: Dict[str, List[float]] = {}
    rank_elapsed: Dict[Any, float] = defaultdict(float)
    ttft_vals: List[float] = []
    sample_kv_mb: List[float] = []

    for metric in metrics:
        token_val = metric.get("total_gen_tokens")
        latency_val = metric.get("total_elapsed_time")
        speed_val = metric.get("avg_speed")
        requests_val = metric.get("total_requests")
        if requests_val is None:
            requests_val = metric.get("request_count")
        if requests_val is None:
            requests_val = metric.get("num_requests")

        if isinstance(token_val, Number):
            total_gen_tokens += float(token_val)
        if isinstance(latency_val, Number):
            total_elapsed_time += float(latency_val)
            rank_elapsed[metric.get("rank", 0)] += float(latency_val)
        if isinstance(speed_val, Number):
            avg_speed_vals.append(float(speed_val))
        if isinstance(requests_val, Number):
            total_requests += float(requests_val)

        ttft_val = metric.get("ttft")
        if isinstance(ttft_val, Number):
            ttft_vals.append(float(ttft_val))

        per_kv = metric.get("per_sample_kv_cache_mb")
        if isinstance(per_kv, list) and per_kv:
            sample_kv_mb.extend(float(x) for x in per_kv if isinstance(x, Number))

        for key, value in metric.items():
            if key in _SKIP_AVG_KEYS:
                continue
            if isinstance(value, Number):
                additional_numeric.setdefault(key, []).append(float(value))

    wall_time = max(rank_elapsed.values()) if rank_elapsed else total_elapsed_time
    summary: Dict[str, Any] = {
        "total_gen_tokens": int(total_gen_tokens) if total_gen_tokens.is_integer() else total_gen_tokens,
        "total_elapsed_time": total_elapsed_time,
        "avg_speed": (
            (total_gen_tokens / total_elapsed_time)
            if total_elapsed_time > 0
            else (sum(avg_speed_vals) / len(avg_speed_vals) if avg_speed_vals else 0.0)
        ),
    }

    if total_requests > 0:
        summary["total_requests"] = int(total_requests) if float(total_requests).is_integer() else total_requests
        summary["avg_latency"] = total_elapsed_time / total_requests
        if wall_time > 0:
            summary["samples_per_second"] = total_requests / wall_time

    if ttft_vals:
        summary["avg_ttft"] = sum(ttft_vals) / len(ttft_vals)

    if sample_kv_mb:
        summary["avg_kv_cache_mb"] = sum(sample_kv_mb) / len(sample_kv_mb)

    for key, values in additional_numeric.items():
        if values:
            summary[f"avg_{key}"] = sum(values) / len(values)

    return summary


def summarize_logged_metrics() -> Dict[str, Any]:
    """Aggregate captured throughput metrics for final reporting.

    Definitions (generation wall time only; preprocess/video decode excluded):
      - samples_per_second: total requests / max(per-rank sum of e2e)
      - avg_ttft: mean time to first token over generate calls
      - avg_kv_cache_mb: mean over samples of full-attention KV bytes after prefill
      - by_task: same aggregates partitioned by task name
    """
    if not _THROUGHPUT_METRICS_HISTORY:
        return {}

    summary = _aggregate_metric_list(_THROUGHPUT_METRICS_HISTORY)
    by_task: Dict[str, Any] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for metric in _THROUGHPUT_METRICS_HISTORY:
        task = metric.get("task")
        if isinstance(task, str) and task:
            grouped[task].append(metric)
        elif isinstance(task, (list, tuple)):
            # Rare: mixed-task batch — attribute whole call to first task name.
            if task and isinstance(task[0], str):
                grouped[task[0]].append(metric)
    for task_name, items in sorted(grouped.items()):
        by_task[task_name] = _aggregate_metric_list(items)
    if by_task:
        summary["by_task"] = by_task
    summary["log_gen_metrics"] = is_gen_metrics_enabled()
    return summary


def _record_metrics(
    total_elapsed_time: float,
    total_gen_tokens: int,
    avg_speed: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "total_elapsed_time": total_elapsed_time,
        "total_gen_tokens": total_gen_tokens,
        "avg_speed": avg_speed,
    }
    if additional_metrics:
        payload.update(additional_metrics)
    _THROUGHPUT_METRICS_HISTORY.append(payload)


def space_tokenizer(text: str) -> float:
    """
    A simple tokenizer that counts the token by the splitted space
    Then a rough estimate of the token count is returned. (times 1.5)
    Args:
        text (str): The input text to tokenize.
    """
    return len(text.split(" ")) * 1.5


def calculate_token_throughput(token_count: float, duration: float) -> float:
    """
    Calculate the token throughput.

    Args:
        token_count (float): The number of tokens processed.
        duration (float): The time taken to process the tokens in seconds.

    Returns:
        float: The token throughput in tokens per second.
    """
    if duration <= 0:
        return 0.0
    return token_count / duration


def log_metrics(
    total_elapsed_time: float,
    total_gen_tokens: int,
    avg_speed: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
):
    """
    Log the metrics in a structured format.

    Args:
        total_elapsed_time (float): Sum of generation latencies in seconds.
        total_gen_tokens (int): The total number of generated tokens.
        avg_speed (float): The average speed in tokens per second.
        additional_metrics (Dict[str, Any]): Additional metrics to log.
    """
    required_stats = f"Metric summary - Total elapsed time: {total_elapsed_time:.3f}s, Total gen tokens: {total_gen_tokens}, Avg speed: {avg_speed:.1f} tokens/s"
    if additional_metrics is not None:
        required_stats += ", Additional metrics: "
        required_stats += ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in additional_metrics.items()
            if k not in {"per_sample_kv_cache_mb"}
        )
    eval_logger.info(required_stats)
    _record_metrics(
        total_elapsed_time=total_elapsed_time,
        total_gen_tokens=total_gen_tokens,
        avg_speed=avg_speed,
        additional_metrics=additional_metrics,
    )


def _make_generation_timing_processor():
    """Build a LogitsProcessor subclass when transformers is available."""
    try:
        from transformers import LogitsProcessor
    except ImportError:
        LogitsProcessor = object  # type: ignore

    class GenerationTimingProcessor(LogitsProcessor):
        """HF LogitsProcessor that records TTFT at the first decode step.

        HuggingFace ``generate`` invokes logits processors after each model forward.
        The first call therefore marks completion of prefill + first-token logits.
        A CUDA synchronize on that first call makes the timestamp GPU-accurate.
        """

        def __init__(self):
            self.t0: Optional[float] = None
            self.ttft_s: Optional[float] = None
            self.num_steps: int = 0
            try:
                import torch

                self._cuda = bool(torch.cuda.is_available())
                self._torch = torch
            except ImportError:
                self._cuda = False
                self._torch = None

        def start(self) -> None:
            """Mark generate start; call after a CUDA sync if using GPU."""
            self.t0 = time.perf_counter()
            self.ttft_s = None
            self.num_steps = 0

        def __call__(self, input_ids, scores):
            self.num_steps += 1
            if self.ttft_s is None and self.t0 is not None:
                if self._cuda and self._torch is not None:
                    self._torch.cuda.synchronize()
                self.ttft_s = time.perf_counter() - self.t0
            return scores

        def finalize(self) -> Tuple[float, Optional[float], int]:
            """Return (e2e_s, ttft_s, num_steps) after generate + CUDA sync."""
            if self.t0 is None:
                return 0.0, None, 0
            if self._cuda and self._torch is not None:
                self._torch.cuda.synchronize()
            e2e = time.perf_counter() - self.t0
            ttft = self.ttft_s if self.ttft_s is not None else e2e
            return e2e, ttft, self.num_steps

    return GenerationTimingProcessor


GenerationTimingProcessor = _make_generation_timing_processor()


def inject_generation_timer(generate_kwargs: Dict[str, Any]) -> GenerationTimingProcessor:
    """Append a timing LogitsProcessor into ``generate_kwargs`` (mutates dict)."""
    timer = GenerationTimingProcessor()
    existing = generate_kwargs.get("logits_processor")
    if existing is None:
        generate_kwargs["logits_processor"] = [timer]
    elif isinstance(existing, list):
        existing.append(timer)
        generate_kwargs["logits_processor"] = existing
    else:
        try:
            generate_kwargs["logits_processor"] = list(existing) + [timer]
        except TypeError:
            generate_kwargs["logits_processor"] = [existing, timer]
    return timer


def kv_cache_bytes(cache) -> Optional[int]:
    """Bytes stored in full-attention K and V. Linear recurrent state is not KV."""
    import torch

    if cache is None:
        return None
    layers = getattr(cache, "layers", None)
    if not layers:
        return None
    total = 0
    found = False
    for layer in layers:
        for name in ("keys", "values", "key_cache", "value_cache"):
            tensor = getattr(layer, name, None)
            if torch.is_tensor(tensor) and tensor.numel() > 0 and tensor.ndim == 4:
                total += int(tensor.numel()) * int(tensor.element_size())
                found = True
    if not found:
        return None
    return total


class _PrefillKvRecorder:
    """Read full-attention KV bytes from the prefill that starts ``generate``."""

    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs
        self.kv_bytes: Optional[int] = None
        self._capture = False
        self._done = False
        self._undo = []

    def _seq_len(self, args, kwargs) -> int:
        import torch

        embeds = kwargs.get("inputs_embeds") if isinstance(kwargs, dict) else None
        ids = kwargs.get("input_ids") if isinstance(kwargs, dict) else None
        if ids is None and args and torch.is_tensor(args[0]) and args[0].ndim == 2:
            ids = args[0]
        if embeds is not None and torch.is_tensor(embeds) and embeds.ndim == 3:
            return int(embeds.shape[1])
        if ids is not None and torch.is_tensor(ids) and ids.ndim == 2:
            return int(ids.shape[-1])
        return 0

    def install(self) -> None:
        root = self.model

        def on_root_pre(module, args, kwargs):
            if self._done:
                return
            self._capture = self._seq_len(args, kwargs) > 1

        def on_root_post(module, args, kwargs, output):
            if not self._capture:
                return
            cache = getattr(output, "past_key_values", None)
            if cache is None and isinstance(output, tuple):
                cache = next((item for item in output if hasattr(item, "layers")), None)
            self.kv_bytes = kv_cache_bytes(cache)
            self._capture = False
            self._done = True

        self._undo.append(root.register_forward_pre_hook(on_root_pre, with_kwargs=True).remove)
        self._undo.append(root.register_forward_hook(on_root_post, with_kwargs=True).remove)

    def close(self) -> None:
        while self._undo:
            self._undo.pop()()

    def kv_cache_mb(self) -> Optional[float]:
        if self.kv_bytes is None:
            return None
        batch = 1
        ids = self.inputs.get("input_ids") if hasattr(self.inputs, "get") else None
        if ids is not None and getattr(ids, "ndim", 0) == 2 and int(ids.shape[0]) > 0:
            batch = int(ids.shape[0])
        return (self.kv_bytes / batch) / (1024.0 * 1024.0)


def measure_prefill_kv(model, inputs, generate_fn):
    """Run ``generate_fn`` and return ``(output, {"kv_cache_mb": ...})``."""
    recorder = _PrefillKvRecorder(model, inputs)
    recorder.install()
    try:
        output = generate_fn()
    finally:
        recorder.close()
    return output, {"kv_cache_mb": recorder.kv_cache_mb()}


def log_hf_generate_metrics(
    *,
    e2e_s: float,
    ttft_s: Optional[float],
    num_steps: int,
    total_gen_tokens: int,
    request_count: int,
    rank: int = 0,
    task: Optional[str] = None,
    per_sample_kv_cache_mb: Optional[List[float]] = None,
) -> None:
    """Record one HF generate call into the run-level metrics history."""
    if not is_gen_metrics_enabled():
        return
    avg_speed = calculate_token_throughput(total_gen_tokens, e2e_s)
    additional: Dict[str, Any] = {
        "request_count": request_count,
        "rank": rank,
        "num_steps": num_steps,
    }
    if ttft_s is not None:
        additional["ttft"] = ttft_s
    if task is not None:
        additional["task"] = task
    if per_sample_kv_cache_mb:
        additional["per_sample_kv_cache_mb"] = [float(x) for x in per_sample_kv_cache_mb]
        additional["kv_cache_mb"] = sum(additional["per_sample_kv_cache_mb"]) / len(additional["per_sample_kv_cache_mb"])
    log_metrics(
        total_elapsed_time=e2e_s,
        total_gen_tokens=total_gen_tokens,
        avg_speed=avg_speed,
        additional_metrics=additional,
    )


class GenMetrics:
    """
    A class to manage the generation of metrics for model evaluation.
    """

    def __init__(self, tokenize_fn: Callable = space_tokenizer):
        self.tokenize_fn = tokenize_fn

    def __enter__(self):
        """
        Initialize the context manager.
        """
        self.metrics = {}
        self.start_time = time.perf_counter()
        return self

    def stop_timer(self):
        self.end_time = time.perf_counter()

    def log_metric(self, content: List[Any], additional_metrics: Optional[Dict[str, Any]] = None):
        num_tokens = sum(self.tokenize_fn(item) for item in content)
        duration = self.end_time - self.start_time
        throughput = calculate_token_throughput(num_tokens, duration)
        self.metrics = {
            "num_tokens": num_tokens,
            "duration": duration,
            "throughput": throughput,
        }
        if additional_metrics:
            self.metrics.update(additional_metrics)

        log_metrics(
            total_elapsed_time=duration,
            total_gen_tokens=int(num_tokens),
            avg_speed=throughput,
            additional_metrics=additional_metrics,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Finalize the context manager and return the collected metrics.
        """
        self.end_time = time.perf_counter()
        self.metrics["duration"] = self.end_time - self.start_time
        return self.metrics

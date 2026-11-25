from functools import wraps
from typing import Optional

import torch
import torch.distributed as dist
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_5 import Qwen3_5

_SAMOD_CLASS_NAME = "Qwen3_5SAMoDForConditionalGeneration"


def _metric_to_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_vision_token_retention(outputs):
    """Return (overall retention, {layer_idx: retention}). Values are in ``[0, 1]``."""
    extra = getattr(outputs, "extra_metrics", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    overall = _metric_to_float(
        extra.get(
            "samod_vision_token_retention",
            getattr(outputs, "samod_vision_token_retention", None),
        )
    )
    layer_rates = {}
    prefix = "samod_vision_token_retention_layer_"
    for key, value in extra.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not suffix.isdigit():
            continue
        rate = _metric_to_float(value)
        if rate is not None:
            layer_rates[int(suffix)] = rate
    return overall, layer_rates


@register_model("qwen3_5_samod")
class Qwen3_5_SAMoD(Qwen3_5):
    """Qwen3.5 SAMoD under the same video eval path as ``qwen3_5``."""

    def _load_pretrained_model(self, pretrained: str, is_moe: bool, attn_implementation: Optional[str]):
        del is_moe
        from verl.models.qwen3_5_samod.samod_runtime import load_samod_model

        model = load_samod_model(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            configure_from_args=False,
        )
        loaded = model.__class__.__name__
        if loaded != _SAMOD_CLASS_NAME:
            raise TypeError(
                f"SAMoD eval requires {_SAMOD_CLASS_NAME}, but {pretrained} loaded {loaded}. "
                "The checkpoint config must set architectures to Qwen3_5SAMoDForConditionalGeneration."
            )
        eval_logger.info(f"Loaded {loaded} from {pretrained} for SAMoD eval")
        return model

    def _generate_and_vision_token_retention(self, inputs, generate_kwargs):
        """Run generate() and record the prefill vision-token retention."""
        batch = int(inputs["input_ids"].shape[0]) if "input_ids" in inputs else 1
        if batch > 1 and not getattr(self, "_samod_vision_token_retention_batch_warned", False):
            eval_logger.warning(
                "SAMoD vision token retention with batch_size>1 is the batch's pooled ratio, "
                "repeated for each sample in the batch."
            )
            self._samod_vision_token_retention_batch_warned = True

        sample_rate = None
        sample_layers = None
        module = self.model
        original_forward = module.forward

        @wraps(original_forward)
        def forward_with_retention(*args, **kwargs):
            nonlocal sample_rate, sample_layers
            outputs = original_forward(*args, **kwargs)
            current_rate, current_layers = _extract_vision_token_retention(outputs)
            if current_rate is not None:
                sample_rate = current_rate
                sample_layers = current_layers
            return outputs

        module.forward = forward_with_retention
        try:
            generated = module.generate(**inputs, **generate_kwargs)
        finally:
            module.forward = original_forward
        return generated, (sample_rate, sample_layers)

    def _empty_retention_bucket(self):
        return {"sum": 0.0, "count": 0, "layer_sums": {}, "layer_counts": {}}

    def _record_sample_vision_token_retentions(self, records) -> None:
        buckets = getattr(self, "_samod_vision_token_retention_local", None)
        if buckets is None:
            buckets = {}
            self._samod_vision_token_retention_local = buckets
        for task_name, observation, is_padding in records:
            if is_padding or not observation or observation[0] is None:
                continue
            rate, layer_rates = observation
            bucket = buckets.setdefault(task_name, self._empty_retention_bucket())
            bucket["sum"] += float(rate)
            bucket["count"] += 1
            for layer_idx, layer_rate in (layer_rates or {}).items():
                bucket["layer_sums"][layer_idx] = bucket["layer_sums"].get(layer_idx, 0.0) + float(layer_rate)
                bucket["layer_counts"][layer_idx] = bucket["layer_counts"].get(layer_idx, 0) + 1
        self.samod_vision_token_retention_by_task = self._sync_vision_token_retentions(buckets)

    def _sync_vision_token_retentions(self, local):
        payload = {
            task: {
                "sum": bucket["sum"],
                "count": bucket["count"],
                "layer_sums": dict(bucket["layer_sums"]),
                "layer_counts": dict(bucket["layer_counts"]),
            }
            for task, bucket in local.items()
        }
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, payload)
        else:
            gathered = [payload]

        merged = {}
        for part in gathered:
            if not part:
                continue
            for task, bucket in part.items():
                acc = merged.setdefault(task, self._empty_retention_bucket())
                acc["sum"] += float(bucket["sum"])
                acc["count"] += int(bucket["count"])
                layer_sums = {int(layer_idx): float(layer_sum) for layer_idx, layer_sum in bucket["layer_sums"].items()}
                layer_counts = {int(layer_idx): int(layer_count) for layer_idx, layer_count in bucket["layer_counts"].items()}
                for layer_idx, layer_sum in layer_sums.items():
                    acc["layer_sums"][layer_idx] = acc["layer_sums"].get(layer_idx, 0.0) + layer_sum
                    acc["layer_counts"][layer_idx] = acc["layer_counts"].get(layer_idx, 0) + layer_counts.get(layer_idx, 0)

        summary = {}
        for task, bucket in sorted(merged.items()):
            if bucket["count"] <= 0:
                continue
            per_layer = {
                str(layer_idx): bucket["layer_sums"][layer_idx] / bucket["layer_counts"][layer_idx]
                for layer_idx in sorted(bucket["layer_sums"])
                if bucket["layer_counts"].get(layer_idx, 0) > 0
            }
            mean_rate = bucket["sum"] / bucket["count"]
            summary[task] = {
                "mean_sample_vision_token_retention": mean_rate,
                "mean_sample_vision_token_retention_pct": mean_rate * 100.0,
                "num_samples": bucket["count"],
                "mean_sample_vision_token_retention_per_layer": per_layer,
            }
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0 and summary:
            for task, item in summary.items():
                eval_logger.info(
                    f"SAMoD vision token retention {task}: "
                    f"{item['mean_sample_vision_token_retention_pct']:.2f}% "
                    f"over {item['num_samples']} samples"
                )
        return summary

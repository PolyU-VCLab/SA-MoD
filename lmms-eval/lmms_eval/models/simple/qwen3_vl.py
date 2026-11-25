import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import inject_generation_timer, is_gen_metrics_enabled, log_hf_generate_metrics, measure_prefill_kv
from lmms_eval.models.model_utils.load_video import _probe_video_metadata

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg")


def _is_video_path(value) -> bool:
    return isinstance(value, str) and value.lower().endswith(_VIDEO_EXTENSIONS)


def cap_nframes(requested: int, total_frames: int) -> int:
    """Largest even frame count that qwen_vl_utils will accept.

    ``smart_nframes`` rounds ``nframes`` to a multiple of 2 and then rejects
    any value above the decoded frame count. A short MVBench clip (for example
    55 frames) cannot take a requested 64, and 55 itself rounds up to 56.
    """
    nframes = min(int(requested), int(total_frames))
    nframes = (nframes // 2) * 2
    if nframes < 2 and total_frames >= 2:
        nframes = 2
    return nframes


def cap_message_nframes(conversations) -> None:
    """Clamp each video's ``nframes`` to the frames that video actually has."""
    for conversation in conversations:
        for message in conversation:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "video" or "nframes" not in item:
                    continue
                video = item.get("video")
                if not _is_video_path(video):
                    continue
                try:
                    total_frames, _ = _probe_video_metadata(video, count_frames=True)
                except Exception as exc:
                    eval_logger.warning(f"Failed to probe video {video}: {exc}")
                    continue
                if total_frames <= 0:
                    continue
                item["nframes"] = cap_nframes(item["nframes"], total_frames)


_NFRAMES_ERROR = re.compile(r"nframes should in interval \[\d+, (\d+)\], but got (\d+)\.")


def _iter_nframe_items(conversations):
    for conversation in conversations:
        for message in conversation:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video" and "nframes" in item:
                    yield item


def read_vision_with_nframes_cap(conversations, reader, **kwargs):
    """Read vision inputs, then shrink ``nframes`` if the decoder has fewer frames.

    The pre-check uses the container frame count. Torchvision or decord can
    decode fewer frames than that header, and ``smart_nframes`` raises. The
    exception carries the decoded count, so retry with that count.
    """
    skipped = set()
    while True:
        try:
            return reader(conversations, **kwargs)
        except ValueError as exc:
            match = _NFRAMES_ERROR.search(str(exc))
            if match is None:
                raise
            total_frames = int(match.group(1))
            target = None
            for item in _iter_nframe_items(conversations):
                if id(item) in skipped or int(item["nframes"]) <= total_frames:
                    continue
                target = item
                break
            if target is None:
                raise
            previous = int(target["nframes"])
            updated = cap_nframes(previous, total_frames)
            if updated < 2 or updated >= previous:
                skipped.add(id(target))
                continue
            target["nframes"] = updated
            eval_logger.warning(f"Video decoded to {total_frames} frames, requested {previous}; sampling {updated}.")
            try:
                return reader(conversations, **kwargs)
            except ValueError as retry_exc:
                retry_match = _NFRAMES_ERROR.search(str(retry_exc))
                if retry_match is None or int(retry_match.group(2)) != previous:
                    raise
                target["nframes"] = previous
                skipped.add(id(target))


def _vision_token_retention_records(requests, sample_vision_token_retentions):
    """Align one vision token retention observation with each request."""
    records = []
    for req, rate in zip(requests, sample_vision_token_retentions):
        metadata = getattr(req, "metadata", None) or {}
        records.append((req.task_name, rate, bool(metadata.get("__padding_only__"))))
    return records


def _resolve_model_class(pretrained: str, is_moe: bool):
    """Auto-detect and return the appropriate HF model class for a Qwen3 variant.

    Returns (model_class, dtype_kwarg_name) where dtype_kwarg_name is the
    keyword argument name for specifying dtype in from_pretrained().
    """
    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if "qwen3_5" in model_type:
        from transformers import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        )

        model_cls = Qwen3_5MoeForConditionalGeneration if is_moe else Qwen3_5ForConditionalGeneration
        dtype_key = "torch_dtype"
    else:
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )

        model_cls = Qwen3VLMoeForConditionalGeneration if is_moe else Qwen3VLForConditionalGeneration
        dtype_key = "dtype"

    return model_cls, dtype_key


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Unified model class for the Qwen3-VL and Qwen3.5 model families.

    Auto-detects the model variant from the HuggingFace config and loads
    the appropriate model class. Supports both dense and MoE variants.

    - Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
    - Qwen3.5:  https://huggingface.co/Qwen/Qwen3.5-4B
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 32 * 32,
        max_pixels: int = 2048 * 32 * 32,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = None,
        force_direct_answer: Optional[bool] = None,
        context_length: Optional[int] = None,
        return_response_json: bool = False,
        strict_generation: bool = False,
        apply_image_pixel_limits: bool = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Auto-detect model variant and load the appropriate HF class.
        # Subclasses override ``_load_pretrained_model`` for custom architectures.
        is_moe = bool(re.search(r"A\d+B", pretrained))
        self._model = self._load_pretrained_model(pretrained, is_moe, attn_implementation).eval()
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.enable_thinking = enable_thinking
        self.force_direct_answer = force_direct_answer
        self.context_length = int(context_length) if context_length is not None else None
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be positive")
        self.return_response_json = return_response_json
        self.strict_generation = str(strict_generation).lower() == "true"
        self.apply_image_pixel_limits = str(apply_image_pixel_limits).lower() == "true"

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _load_pretrained_model(self, pretrained: str, is_moe: bool, attn_implementation: Optional[str]):
        """Load the stock Qwen3-VL / Qwen3.5 class selected from the HF config."""
        model_cls, dtype_key = _resolve_model_class(pretrained, is_moe)
        model_kwargs = {
            dtype_key: "bfloat16",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        return model_cls.from_pretrained(pretrained, **model_kwargs)

    def _generate_and_vision_token_retention(self, inputs, generate_kwargs):
        return self.model.generate(**inputs, **generate_kwargs), None

    def _record_sample_vision_token_retentions(self, records) -> None:
        return None

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Score text continuations with summed NLL, matching task argmin scoring.

        The chat adapter inherits the same likelihood request protocol. Only
        continuation tokens are scored: no prompt loss or assistant EOS token.
        Joint tokenization includes any token that crosses the prompt boundary.
        Media likelihood is intentionally unsupported rather than ignored.
        """
        results = []
        if not requests:
            return results
        for request in tqdm(requests, disable=self.rank != 0, desc="Model Scoring"):
            if len(request.args) == 2:
                context, continuation = request.args
            elif len(request.args) == 6:
                context, continuation, doc_to_visual, doc_id, task, split = request.args
                doc = self.task_dict[task][split][doc_id]
                if doc_to_visual is not None and doc_to_visual(doc):
                    raise NotImplementedError("Qwen3-VL loglikelihood currently supports text-only requests. Use a generative task for image or video inputs.")
                if callable(continuation):
                    continuation = continuation(doc)
            else:
                raise ValueError(f"Expected two or six likelihood arguments, got {len(request.args)}")
            if not isinstance(context, str) or not isinstance(continuation, str):
                raise TypeError("Likelihood context and continuation must be strings")
            if continuation == "":
                results.append((0.0, True))
                continue

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": [{"type": "text", "text": context}]})
            prompt = self._apply_chat_template([messages])[0]
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            full_tokens = self.tokenizer.encode(prompt + continuation, add_special_tokens=False)

            # A BPE token can change when the continuation is appended. Keep
            # the shared prefix, and include the overlapping token in scoring.
            prefix_length = 0
            for prompt_token, full_token in zip(prompt_tokens, full_tokens):
                if prompt_token != full_token:
                    break
                prefix_length += 1
            if prefix_length == 0:
                raise ValueError("Likelihood scoring requires a nonempty tokenized prompt prefix")
            continuation_length = len(full_tokens) - prefix_length
            if continuation_length == 0:
                results.append((0.0, True))
                continue
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    logits_to_keep=continuation_length + 1,
                )
                # Position P-1 predicts the first continuation token at P.
                logits = output.logits[:, :-1, :].float()
                targets = input_ids[:, prefix_length:]
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
                greedy = bool((logits.argmax(dim=-1) == targets).all().item())
            result = (float(loss.item()), greedy)
            results.append(result)
            self.cache_hook.add_partial("loglikelihood", (context, continuation), result)
        return results

    def flatten(self, input):  # noqa: A002 - Preserve the existing keyword argument name.
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_video_kwargs(self):
        """Build video processing kwargs based on model configuration."""
        video_kwargs = {"min_pixels": self.min_pixels}

        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        elif self.total_pixels is not None:
            video_kwargs["max_frames"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames

        if self.total_pixels is not None:
            video_kwargs["total_pixels"] = self.total_pixels
        else:
            video_kwargs["max_pixels"] = self.max_pixels

        return video_kwargs

    def _apply_chat_template(self, batched_messages, **kwargs):
        """Apply chat template with optional enable_thinking support."""
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        if self.force_direct_answer is not None:
            template_kwargs["force_direct_answer"] = self.force_direct_answer
        template_kwargs.update(kwargs)
        return self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True, **template_kwargs)

    def _build_generate_kwargs(self, gen_kwargs):
        """Build model.generate() kwargs from user gen_kwargs merged with defaults."""
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        pad_token_id = self.tokenizer.pad_token_id

        if current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)

        generate_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": current["do_sample"],
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            val = current.get(key)
            if val is not None:
                generate_kwargs[key] = val

        return generate_kwargs

    def _strip_thinking(self, answer):
        """Strip <think>...</think> content from model output if enable_thinking is set."""
        if self.enable_thinking:
            _, _, remaining = answer.partition("</think>")
            return remaining.strip()
        return answer

    def _check_context(self, inputs, max_new_tokens):
        if not getattr(self, "strict_generation", False) and getattr(self, "context_length", None) is None:
            return
        config = self.model.config
        text_config = getattr(config, "text_config", config)
        limits = [
            getattr(text_config, "max_position_embeddings", None),
            getattr(self, "context_length", None),
        ]
        limits = [int(limit) for limit in limits if limit is not None and int(limit) > 0]
        input_tokens = int(inputs["input_ids"].shape[-1])
        if limits and input_tokens + max_new_tokens > min(limits):
            raise ValueError(f"Context capacity exceeded: {input_tokens} input + {max_new_tokens} output > {min(limits)} total tokens. Increase the supported context capacity; frames and output budget have not been reduced.")

    def _format_response(self, raw, input_tokens, output_tokens, max_new_tokens):
        final = self._strip_thinking(raw)
        if not getattr(self, "return_response_json", False):
            return final
        import json

        return json.dumps(
            {
                "api_response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": final, "raw_generation": raw}, "finish_reason": "length" if output_tokens >= max_new_tokens else "stop"}],
                        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
                    },
                }
            },
            ensure_ascii=False,
        )

    def _preprocess_chunk(self, chunk):
        """Preprocess a batch chunk on CPU: message building, video decoding, tokenization.

        Returns (inputs, contexts, gen_kwargs, until) with inputs still on CPU.
        """
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
        until = [item for item in until if item != "\n\n"]

        if isinstance(contexts, tuple):
            contexts = list(contexts)

        for i in range(len(contexts)):
            if "<image>" in contexts[i]:
                contexts[i] = contexts[i].replace("<image>", "")

        video_kwargs = self._build_video_kwargs()

        batched_messages = []
        for i, context in enumerate(contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            message = [{"role": "system", "content": self.system_prompt}]

            if self.reasoning_prompt:
                context = context.strip() + self.reasoning_prompt
                contexts[i] = context

            processed_visuals = []
            if visual_list[i] is not None:
                for visual in visual_list[i]:
                    if _is_video_path(visual):
                        # Cap nframes to actual video frame count to avoid ValueError
                        # when video has fewer frames than max_num_frames
                        per_video_kwargs = {**video_kwargs}
                        if "nframes" in per_video_kwargs:
                            total_frames, _ = _probe_video_metadata(visual, count_frames=True)
                            per_video_kwargs["nframes"] = cap_nframes(per_video_kwargs["nframes"], total_frames)
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": visual,
                                **per_video_kwargs,
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )

            if self.interleave_visuals is False:
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
            else:
                image_placeholders = re.findall(r"<image \d+>", context)
                content_parts = []
                text_parts = re.split(r"<image \d+>", context)
                if text_parts[0]:
                    content_parts.append({"type": "text", "text": text_parts[0]})

                for placeholder_idx, placeholder in enumerate(image_placeholders):
                    img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                    image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                    if processed_visuals and image_idx < len(processed_visuals):
                        content_parts.append(processed_visuals[image_idx])
                    if placeholder_idx + 1 < len(text_parts) and text_parts[placeholder_idx + 1]:
                        content_parts.append({"type": "text", "text": text_parts[placeholder_idx + 1]})

                message.append(
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                )

            batched_messages.append(message)

        texts = self._apply_chat_template(batched_messages)
        image_inputs, video_inputs, processed_video_kwargs = read_vision_with_nframes_cap(
            batched_messages,
            process_vision_info,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        video_metadata_list = None
        if video_inputs is not None:
            video_inputs, video_metadata_list = map(list, zip(*video_inputs))

        if self.batch_size > 1:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )

        return inputs, contexts, gen_kwargs, until

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        sample_vision_token_retentions = []
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._preprocess_chunk, chunks[0]) if chunks else None

            for idx in range(len(chunks)):
                inputs, contexts, gen_kwargs, until = future.result()

                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                self._check_context(inputs, generate_kwargs["max_new_tokens"])
                collect_gen_metrics = is_gen_metrics_enabled()
                timer = inject_generation_timer(generate_kwargs) if collect_gen_metrics else None
                if timer is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    timer.start()
                prefill_stats = None
                if collect_gen_metrics:
                    (cont, sample_vision_token_retention), prefill_stats = measure_prefill_kv(
                        self.model,
                        inputs,
                        lambda: self._generate_and_vision_token_retention(inputs, generate_kwargs),
                    )
                else:
                    cont, sample_vision_token_retention = self._generate_and_vision_token_retention(inputs, generate_kwargs)
                e2e_s = ttft_s = None
                num_steps = 0
                if timer is not None:
                    e2e_s, ttft_s, num_steps = timer.finalize()

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                if timer is not None and e2e_s is not None:
                    batch = int(inputs["input_ids"].shape[0])
                    kv_each = None if prefill_stats is None else prefill_stats.get("kv_cache_mb")
                    log_hf_generate_metrics(
                        e2e_s=e2e_s,
                        ttft_s=ttft_s,
                        num_steps=num_steps,
                        total_gen_tokens=sum(len(ids) for ids in generated_ids_trimmed),
                        request_count=batch,
                        rank=self.rank,
                        per_sample_kv_cache_mb=None if kv_each is None else [kv_each] * batch,
                    )
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for i, (ans, context) in enumerate(zip(answers, contexts)):
                    ans = self._format_response(
                        ans,
                        int(inputs["input_ids"].shape[-1]),
                        len(generated_ids_trimmed[i]),
                        generate_kwargs["max_new_tokens"],
                    )
                    res.append(ans)
                    sample_vision_token_retentions.append(sample_vision_token_retention)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        sample_vision_token_retentions = re_ords.get_original(sample_vision_token_retentions)
        self._record_sample_vision_token_retentions(_vision_token_retention_records(requests, sample_vision_token_retentions))
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
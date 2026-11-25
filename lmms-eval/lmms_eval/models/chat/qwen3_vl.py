import time
from typing import List

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import inject_generation_timer, is_gen_metrics_enabled, log_hf_generate_metrics, log_metrics, measure_prefill_kv
from lmms_eval.models.chat.frame_sequence import apply_frame_timestamps, pack_timestamped_frames
from lmms_eval.models.simple.qwen3_vl import _vision_token_retention_records, cap_message_nframes, read_vision_with_nframes_cap
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0
        sample_vision_token_retentions = []
        collect_gen_metrics = is_gen_metrics_enabled()

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            batch_task = task[0] if task else None

            chat_messages: List[ChatMessages] = []
            visuals = []
            videos = []
            for idx, (ids, task_name, split_name) in enumerate(zip(doc_id, task, split)):
                messages = doc_to_messages[idx](self.task_dict[task_name][split_name][ids])
                messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
                chat_message = ChatMessages(**{"messages": messages})
                visual, video, _ = chat_message.extract_media()
                visuals.append(visual)
                videos.append(video)
                chat_messages.append(chat_message)

            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = self._build_video_kwargs()
            image_kwargs = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels} if getattr(self, "apply_image_pixel_limits", False) else {}
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs, image_kwargs=image_kwargs) for chat_message in chat_messages]
            frame_timestamps = pack_timestamped_frames(batched_messages, video_kwargs)
            cap_message_nframes(batched_messages)

            texts = self._apply_chat_template(batched_messages)

            image_inputs, video_inputs, video_kwargs_qwen = read_vision_with_nframes_cap(
                batched_messages,
                process_vision_info,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_kwargs = {**video_kwargs, **video_kwargs_qwen}

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
                apply_frame_timestamps(video_inputs, video_metadatas, frame_timestamps)

            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
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
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    return_tensors="pt",
                )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            generate_kwargs = self._build_generate_kwargs(gen_kwargs)
            self._check_context(inputs, generate_kwargs["max_new_tokens"])

            timer = None
            if collect_gen_metrics:
                timer = inject_generation_timer(generate_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.start()
            start_time = time.time()
            prefill_stats = None
            if collect_gen_metrics:
                (cont, sample_vision_token_retention), prefill_stats = measure_prefill_kv(
                    self.model,
                    inputs,
                    lambda: self._generate_and_vision_token_retention(inputs, generate_kwargs),
                )
            else:
                cont, sample_vision_token_retention = self._generate_and_vision_token_retention(inputs, generate_kwargs)
            end_time = time.time()
            e2e_s = ttft_s = None
            num_steps = 0
            if timer is not None:
                e2e_s, ttft_s, num_steps = timer.finalize()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            if collect_gen_metrics and e2e_s is not None:
                batch = max(len(texts), 1)
                kv_each = None if prefill_stats is None else prefill_stats.get("kv_cache_mb")
                log_hf_generate_metrics(
                    e2e_s=e2e_s,
                    ttft_s=ttft_s,
                    num_steps=num_steps,
                    total_gen_tokens=sum(len(ids) for ids in generated_ids_trimmed),
                    request_count=batch,
                    rank=self.rank,
                    task=batch_task,
                    per_sample_kv_cache_mb=None if kv_each is None else [kv_each] * batch,
                )

            for i, (ans, context) in enumerate(zip(answers, texts)):
                input_tokens = int(inputs["input_ids"].shape[-1])
                output_tokens = len(generated_ids_trimmed[i])
                ans = self._format_response(ans, input_tokens, output_tokens, generate_kwargs["max_new_tokens"])
                reported_input_tokens = input_tokens if getattr(self, "return_response_json", False) else None
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(input_tokens=reported_input_tokens, output_tokens=output_tokens)))
                sample_vision_token_retentions.append(sample_vision_token_retention)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")
            pbar.update(1)

        res = re_ords.get_original(res)
        sample_vision_token_retentions = re_ords.get_original(sample_vision_token_retentions)
        self._record_sample_vision_token_retentions(_vision_token_retention_records(requests, sample_vision_token_retentions))

        if not collect_gen_metrics:
            avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
            log_metrics(
                total_gen_tokens=total_tokens,
                total_elapsed_time=total_elapsed_time,
                avg_speed=avg_speed,
                additional_metrics={"rank": self.rank},
            )

        pbar.close()
        return res
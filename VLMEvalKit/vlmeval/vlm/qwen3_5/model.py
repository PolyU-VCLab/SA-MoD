from __future__ import annotations

import logging
import os
import warnings
from functools import wraps

import torch

from ..base import BaseModel
from .prompt import Qwen35VLPromptMixin


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen35VLChat(Qwen35VLPromptMixin, BaseModel):
    """Qwen3.5 SAMoD chat model for VLMEvalKit."""

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 4096,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        attn_implementation: str | None = 'flash_attention_2',
        model_choice: str = 'samod',
        chat_template_kwargs: dict | None = None,
        model_choice_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        if model_choice not in ('samod', 'original'):
            raise ValueError(
                f'Qwen3.5 eval model_choice must be samod or original, got {model_choice!r}'
            )
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.chat_template_kwargs = self._load_chat_template_kwargs(chat_template_kwargs)
        self.model_choice_kwargs = model_choice_kwargs or {}
        self.model_choice = model_choice
        self.report_vision_token_retention = bool(kwargs.pop('report_vision_token_retention', False))
        self._vlmeval_vision_token_retention_sum = 0.0
        self._vlmeval_vision_token_retention_num_samples = 0
        self._vlmeval_vision_token_retention_values: list[float] = []
        self._vlmeval_vision_token_retention_layer_sums: dict[int, float] = {}
        self._vlmeval_vision_token_retention_layer_counts: dict[int, int] = {}
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2

        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.use_vllm = False
        if kwargs.get('use_vllm', False):
            raise ValueError('Qwen3.5 eval uses the transformers generate path, not vLLM')

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if model_choice == 'original':
            from transformers import Qwen3_5ForConditionalGeneration

            model_kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map='auto',
                trust_remote_code=True,
            )
            if attn_implementation is not None:
                model_kwargs['attn_implementation'] = attn_implementation
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            self.model.eval()
            print('Using model choice: original')
            torch.cuda.empty_cache()
            return

        from verl.models.qwen3_5_samod.samod_runtime import load_samod_model

        samod_kwargs = dict(self.model_choice_kwargs or {})
        from_scratch = bool(
            samod_kwargs.pop('from_scratch', samod_kwargs.pop('samod_from_scratch', False))
        )
        self.model = load_samod_model(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            configure_from_args=from_scratch,
            samod_kwargs=samod_kwargs if from_scratch else None,
        )
        self.model.eval()
        print(
            'Using model choice: samod '
            f'(from_scratch={from_scratch}, report_vision_token_retention={self.report_vision_token_retention})'
        )
        torch.cuda.empty_cache()

    def _load_chat_template_kwargs(self, chat_template_kwargs: dict | None) -> dict:
        template_kwargs = dict(chat_template_kwargs or {})
        chat_template_path = template_kwargs.pop('chat_template_path', None)
        chat_template = template_kwargs.get('chat_template', chat_template_path)

        if isinstance(chat_template, str) and os.path.isfile(chat_template):
            with open(chat_template, encoding='utf-8') as f:
                template_kwargs['chat_template'] = f.read()
        elif chat_template_path is not None:
            raise FileNotFoundError(f'Chat template file not found: {chat_template_path}')
        return template_kwargs

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {'type': 'video', 'video': [ensure_image_url(v) for v in value]}
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        item['nframes'] = min(self.nframe, frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _build_messages(self, message, dataset=None) -> list[dict]:
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')
        return messages

    def _apply_chat_template(self, messages: list[dict]) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def _post_process_response(self, response: str) -> str:
        if getattr(self, 'post_process', False):
            resp = response.split('\\boxed{')[-1]
            counter, end = 1, None
            for i, char in enumerate(resp):
                if char == '{':
                    counter += 1
                elif char == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
            if end is not None:
                response = resp[:end]

        template_kwargs = getattr(self, 'chat_template_kwargs', {})
        if template_kwargs.get('enable_thinking', False) is False or \
            template_kwargs.get('force_direct_answer', False) is True:
            if '</think>' in response:
                return response.rsplit('</think>', 1)[-1].strip()
            if '<think>' in response:
                return response.rsplit('<think>', 1)[0].strip()
            return response.strip()
        return response

    @staticmethod
    def _metric_to_float(value) -> float | None:
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

    def _extract_vision_token_retention(self, outputs) -> tuple[float | None, dict[int, float]]:
        """Return (overall retention, {layer_idx: retention}). Values are in ``[0, 1]``."""
        extra_metrics = getattr(outputs, 'extra_metrics', None) or {}
        overall = self._metric_to_float(
            extra_metrics.get(
                'samod_vision_token_retention',
                getattr(outputs, 'samod_vision_token_retention', None),
            )
        )

        layer_rates: dict[int, float] = {}
        prefix = 'samod_vision_token_retention_layer_'
        for key, value in extra_metrics.items():
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if not suffix.isdigit():
                continue
            rate = self._metric_to_float(value)
            if rate is not None:
                layer_rates[int(suffix)] = rate
        return overall, layer_rates

    def _record_vision_token_retention(self, retention: float | None, layer_rates: dict[int, float] | None = None) -> None:
        if retention is not None:
            rate = float(retention)
            self._vlmeval_vision_token_retention_sum += rate
            self._vlmeval_vision_token_retention_num_samples += 1
            self._vlmeval_vision_token_retention_values.append(rate)
        if not layer_rates:
            return
        for layer_idx, rate in layer_rates.items():
            self._vlmeval_vision_token_retention_layer_sums[layer_idx] = (
                self._vlmeval_vision_token_retention_layer_sums.get(layer_idx, 0.0) + float(rate)
            )
            self._vlmeval_vision_token_retention_layer_counts[layer_idx] = (
                self._vlmeval_vision_token_retention_layer_counts.get(layer_idx, 0) + 1
            )

    def _generate_with_vision_token_retention(self, inputs):
        if not self.report_vision_token_retention:
            return self.model.generate(**inputs, **self.generate_kwargs)

        original_forward = self.model.forward
        sample_retention = None
        sample_layer_rates: dict[int, float] | None = None

        @wraps(original_forward)
        def forward_with_retention(*args, **kwargs):
            nonlocal sample_retention, sample_layer_rates
            outputs = original_forward(*args, **kwargs)
            current_rate, current_layers = self._extract_vision_token_retention(outputs)
            if current_rate is not None:
                sample_retention = current_rate
                sample_layer_rates = current_layers
            return outputs

        self.model.forward = forward_with_retention
        try:
            return self.model.generate(**inputs, **self.generate_kwargs)
        finally:
            self.model.forward = original_forward
            self._record_vision_token_retention(sample_retention, sample_layer_rates)

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_transformers(message, dataset=dataset)

    def generate_inner_transformers(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("Please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = self._build_messages(message, dataset=dataset)
        text = self._apply_chat_template(messages)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors='pt',
            **(video_kwargs or {}),
        )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        generated_ids = self._generate_with_vision_token_retention(inputs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        response = self._post_process_response(response)
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response


Qwen3_5VLChat = Qwen35VLChat

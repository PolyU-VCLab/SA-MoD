import copy
import fcntl
import json
import logging
import os
import os.path as osp
import string
import warnings
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE
from vlmeval.smp import cn_string, encode_image_to_base64, splitlen
from ..base import BaseModel


class LLaVA_DCT(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="liuhaotian/llava_v1.5_7b", **kwargs):
        try:
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical(
                "Please install llava from https://github.com/haotian-liu/LLaVA"
            )
            raise err

        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        if model_path == "Lin-Chen/ShareGPT4V-7B":
            model_name = "llava-v1.5-7b"
        elif model_path == "Lin-Chen/ShareGPT4V-13B":
            model_name = "llava-v1.5-13b"
        else:
            model_name = get_model_name_from_path(model_path)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    device_map="cpu",
                )
            )
        except Exception as err:
            if "ShareGPT4V" in model_path:
                import llava

                logging.critical(
                    "Please manually remove the encoder type check in "
                    f"{llava.__path__[0]}/model/multimodal_encoder/builder.py "
                    "Line 8 to use the ShareGPT4V model. "
                )
            else:
                logging.critical("Unknown error when loading LLaVA model.")
            raise err

        self.model = self.model.cuda()
        self.conv_mode = "llava_v1"
        self.saliency_threshold = kwargs.pop("saliency_threshold", 0.5)
        self.output_token_avg_kl_dir = kwargs.pop("output_token_avg_kl_dir", None)
        self.output_token_avg_kl_records = {}

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    @staticmethod
    def lowpass_filter(img, kernel_size=11, sigma=5.0):
        channels, _, _ = img.shape
        device = img.device
        dtype = img.dtype
        ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(
            img.unsqueeze(0),
            kernel,
            padding=kernel_size // 2,
            groups=channels,
        ).squeeze(0)

    def perturb_image_tensor(self, image_tensor):
        if image_tensor is None:
            return None
        if image_tensor.ndim == 4:
            return torch.stack([self.lowpass_filter(img) for img in image_tensor], dim=0)
        if image_tensor.ndim == 3:
            return self.lowpass_filter(image_tensor)
        raise ValueError(f"Unsupported image tensor shape: {tuple(image_tensor.shape)}")

    def record_output_token_avg_kl(self, dataset, token_kl, token_avg_kl):
        dataset_name = dataset or "unknown_dataset"
        token_kl_values = token_kl.detach().float().cpu().reshape(-1).tolist()
        kl_value = float(token_avg_kl.detach().cpu().item())
        records = self.output_token_avg_kl_records.setdefault(dataset_name, [])
        records.append(kl_value)

        output_dir = self.output_token_avg_kl_dir or os.environ.get(
            "VLMEVAL_RELATIVE_PRIOR_AVG_KL_DIR"
        )
        if output_dir is None:
            return

        os.makedirs(output_dir, exist_ok=True)
        rank = (
            os.environ.get("RANK")
            or os.environ.get("LOCAL_RANK")
            or os.environ.get("SLURM_PROCID")
            or "0"
        )
        worker_id = f"rank{rank}_pid{os.getpid()}"
        jsonl_path = osp.join(
            output_dir,
            f"relative_prior_avg_kl_{worker_id}.jsonl",
        )
        record = {
            "dataset": dataset_name,
            "worker": worker_id,
            "local_index": len(records),
            "output_token_kl": token_kl_values,
            "output_token_avg_kl": kl_value,
        }
        lock_path = osp.join(
            output_dir,
            "relative_prior_avg_kl.lock",
        )
        with open(lock_path, "w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                values = []
                token_count = 0
                token_sum = 0.0
                prefix = "relative_prior_avg_kl_"
                for filename in os.listdir(output_dir):
                    if not filename.startswith(prefix) or not filename.endswith(".jsonl"):
                        continue
                    path = osp.join(output_dir, filename)
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                saved_record = json.loads(line)
                                values.append(saved_record["output_token_avg_kl"])
                                saved_token_kl = saved_record.get("output_token_kl", [])
                                token_count += len(saved_token_kl)
                                token_sum += float(np.sum(saved_token_kl))
                            except (json.JSONDecodeError, KeyError):
                                continue

                summary = {
                    "dataset": dataset_name,
                    "count": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "token_count": token_count,
                    "token_mean": token_sum / token_count if token_count else None,
                    "jsonl_prefix": prefix,
                }
                summary_path = osp.join(
                    output_dir,
                    "relative_prior_avg_kl_summary.json",
                )
                tmp_path = f"{summary_path}.tmp.{worker_id}"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4, ensure_ascii=False)
                os.replace(tmp_path, summary_path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def chat_inner(self, message, dataset=None):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += "USER: " if utter["role"] == "user" else "ASSISTANT: "
            content, images_sub = self.concat_tilist(utter["content"])
            prompt += content
            images.extend(images_sub)
            prompt += " " if utter["role"] == "user" else self.stop_str
        assert message[-1]["role"] == "user", message
        prompt += "ASSISTANT: "

        images = [Image.open(s).convert("RGB") for s in images]
        args = Namespace()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, self.image_processor, args).to(
            "cuda", dtype=torch.float16
        )

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                stopping_criteria=[stopping_criteria],
                **self.kwargs,
            )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return output

    def generate_inner(self, message, dataset=None):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert("RGB") for s in images]
        args = Namespace()
        args.image_aspect_ratio = "pad"
        if images:
            image_tensor = process_images(images, self.image_processor, args).to(
                "cuda", dtype=torch.float16
            )
        else:
            image_tensor = None
        
        content = content + (
            " Please reason step by step before giving the final answer."
        )

        prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.enable_grad():
            with torch.no_grad():
                generate_kwargs = {
                    **self.kwargs,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                generation_output = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    stopping_criteria=[stopping_criteria],
                    **self.kwargs,
                )
                output_ids = generation_output.sequences
                output_logits = torch.stack(generation_output.scores, dim=1)
            forward_input_ids = torch.cat([input_ids, output_ids], dim=1)

            def remove_image_tokens(module, inputs, output):
                if output.ndim == 3:
                    return output[:, :0, :]
                if output.ndim == 2:
                    return output[:0, :]
                raise ValueError(
                    f"Unsupported projector output shape: {tuple(output.shape)}"
                )

            handle = None
            if image_tensor is not None:
                handle = self.model.get_model().mm_projector.register_forward_hook(
                    remove_image_tokens
                )
            try:
                with torch.no_grad():
                    output_logits_no_image = self.model(
                        input_ids=forward_input_ids,
                        images=image_tensor,
                        use_cache=False,
                        return_dict=True,
                    ).logits.detach()
            finally:
                if handle is not None:
                    handle.remove()
            output_logits_no_image = output_logits_no_image[:, -output_logits.shape[1]-1:-1]
            output_token_kl = F.kl_div(
                F.log_softmax(output_logits, dim=-1),
                F.softmax(output_logits_no_image, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            output_token_avg_kl = output_token_kl.mean()
            self.record_output_token_avg_kl(
                dataset, output_token_kl, output_token_avg_kl
            )

        
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output


class LLaVA_Next(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="llava-hf/llava-v1.6-vicuna-7b-hf", **kwargs):
        from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                                  LlavaNextForConditionalGeneration, LlavaNextProcessor)

        self.model_path = model_path
        if "34b" in model_path.lower():
            self.processor = LlavaNextProcessor.from_pretrained(
                self.model_path, use_fast=False
            )
        elif "interleave" in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        flash_attn_flag = False
        try:
            import flash_attn  # noqa: F401

            flash_attn_flag = True
        except ImportError:
            pass

        if flash_attn_flag:
            if "interleave" in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,
                )
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,
                )
        else:
            if "interleave" in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )

        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=2048, top_p=None, num_beams=1
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def apply_prompt_template(self, prompt):
        model_path = self.model_path.lower()
        if "mistral" in model_path:
            template = "[INST] PLACEHOLDER [/INST]"
        elif "vicuna" in model_path:
            template = (
                "A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                "USER: PLACEHOLDER ASSISTANT:"
            )
        elif "34b" in model_path:
            template = (
                "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\nPLACEHOLDER<|im_end|>"
                "<|im_start|>assistant\n"
            )
        else:
            raise NotImplementedError(
                f"Prompt template for {model_path} not implemented."
            )

        prompt = template.replace("PLACEHOLDER", f"<image>\n{prompt}")
        return prompt

    def output_process(self, answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        content, images = [], []
        for msg in message:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": msg["value"]})
            else:
                content.append({"type": "image"})
                images.append(Image.open(msg["value"]).convert("RGB"))
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(prompt, images, return_tensors="pt").to(
            "cuda", torch.float16
        )
        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        answer = self.output_process(answer)
        answer = answer.replace('<unk>', '')
        return answer


class LLaVA_Next2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="lmms-lab/llama3-llava-next-8b", **kwargs):
        assert model_path is not None
        try:
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                                        tokenizer_image_token)
            from llava.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map=None
        )
        model.cuda().eval()
        model.tie_weights()

        if "llama3" in model_path.lower():
            conv_mode = "llava_llama_3"
        elif "qwen" in model_path.lower():
            conv_mode = "qwen_1_5"
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner(self, message, dataset=None):
        content, images = "", []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                images.append(Image.open(msg["value"]).convert("RGB"))
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        preprocess = self.image_processor.preprocess
        image_tokenizer = self.tokenizer_image_token
        image_tensor = [
            preprocess(f, return_tensors="pt")["pixel_values"][0].half().cuda()
            for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = image_tokenizer(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs


class LLaVA_OneVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="lmms-lab/llava-onevision-qwen2-7b-si", **kwargs):
        assert model_path is not None
        try:
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                                        process_images, tokenizer_image_token)
            from llava.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        video_kwargs_default = dict(
            overwrite=True, mm_spatial_pool_mode="average", force_sample=True
        )
        video_kwargs_default.update(kwargs)
        self.video_kwargs = video_kwargs_default

        overwrite_config = None
        if "video" in model_path.lower():
            if self.video_kwargs["overwrite"]:
                overwrite_config = {}
                overwrite_config["mm_spatial_pool_mode"] = self.video_kwargs[
                    "mm_spatial_pool_mode"
                ]

        model_name = get_model_name_from_path(model_path)
        import warnings

        # filter warning align with official code
        warnings.filterwarnings("ignore")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="auto",
            overwrite_config=overwrite_config,
        )
        model.eval()
        model.tie_weights()

        if "llava" in model_path.lower():
            conv_mode = "qwen_1_5"
        if 'llava-video' in model_path.lower():
            self.nframe = 64
        else:
            self.nframe = 16
            if "72b" in model_path.lower():
                self.nframe = 32

        if "video" in model_path.lower():
            self.force_sample = self.video_kwargs["force_sample"]
        else:
            self.force_sample = False

        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = (
            process_images  # Store process_images as a class attribute
        )
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(
            images, self.image_processor, self.model.config
        )
        image_tensor = [
            _image.to(dtype=torch.float16, device="cuda") for _image in image_tensor
        ]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            else:
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError(
                "LLaVA-OneVision does not support multiple videos as input."
            )

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, 1, self.force_sample
        )

        time_instruciton = (
            f"The video lasts for {video_time:.2f} seconds,"
            f"and {len(video_frames[0])} frames are uniformly sampled from it."
            f"These frames are located at {frame_time}."
            f"Please answer the following questions related to this video.\n"
        )

        if self.force_sample:
            content = visual_content + time_instruciton + text_content
        else:
            content = visual_content + text_content

        image_tensors = []
        frames = (
            self.image_processor.preprocess(video_frames, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )
        image_tensors.append(frames)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()
        image_sizes = [frame.size for frame in video_frames]
        modalities = ["video"] * len(video_frames)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            modalities=modalities,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu

        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, sample_fps, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()
        return spare_frames, frame_time, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)


class LLaVA_OneVision_HF(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", **kwargs):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        assert model_path is not None, "Model path must be provided."
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to('cuda')
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.video_kwargs = kwargs.get("video_kwargs", {})
        self.force_sample = self.video_kwargs.get("force_sample", False)
        self.nframe = kwargs.get("nframe", 8)
        self.fps = 1
        self.model_path = model_path

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            elif msg["type"] == "video":
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError("LLaVA-OneVision does not support multiple videos as input.")

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, fps=1, force_sample=self.force_sample
        )

        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, "
            f"and {len(video_frames)} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video.\n"
        )

        content = visual_content + time_instruction + text_content
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": content}, {"type": "video"}],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(videos=video_frames, text=prompt, return_tensors="pt").to('cuda', torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()

        if avg_fps == 0:
            raise ValueError(f"Video '{video_path}' has an average FPS of 0, which is invalid.")
        if fps <= 0:
            raise ValueError("FPS argument must be greater than 0.")

        effective_fps = round(avg_fps / fps)
        frame_idx = list(range(0, total_frame_num, effective_fps))
        frame_time = [i / avg_fps for i in frame_idx]

        if len(frame_idx) > max_frames_num or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / avg_fps for i in frame_idx]

        frame_time_str = ", ".join([f"{t:.2f}s" for t in frame_time])
        video_frames = vr.get_batch(frame_idx).asnumpy()
        video_time = total_frame_num / avg_fps

        return video_frames, frame_time_str, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == "VIDEO" and 'megabench' not in dataset.lower():
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)


class LLaVA_OneVision_1_5(BaseModel):
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self, model_path="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct", **kwargs):
        from transformers import AutoModelForCausalLM, AutoProcessor

        assert model_path is not None, "Model path must be provided."
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype='auto', trust_remote_code=True).to('cuda')
        self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.model_path = model_path
        kwargs.setdefault('max_new_tokens', 4096)
        self.img_size = kwargs.pop('img_size', -1)
        self.kwargs = kwargs

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    content_list.append(dict(type='image_url', image_url=f'data:image/jpeg;base64,{b64}'))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, dataset=None):
        from qwen_vl_utils import process_vision_info

        message = self.prepare_inputs(inputs)
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=1280 * 28 * 28 // len(image_inputs),
            max_pixels=32768 * 28 * 28 // len(image_inputs),
        )
        print(f'pixel_values.shape: {inputs["pixel_values"].shape}')
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

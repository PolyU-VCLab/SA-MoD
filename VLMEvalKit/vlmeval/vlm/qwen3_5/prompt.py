from __future__ import annotations


class Qwen35VLPromptMixin:
    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {"SSI_Bench"}:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        return dataset_type in {'MCQ', 'Y/N', 'VQA'}

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)

        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _image_messages(self, line, dataset: str) -> list[dict[str, str]]:
        tgt_path = self.dump_image(line, dataset)
        if isinstance(tgt_path, list):
            return [dict(type='image', value=p) for p in tgt_path]
        return [dict(type='image', value=tgt_path)]

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        import string

        import pandas as pd

        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}

        prompt = ''
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if options:
            prompt += 'Options:\n'
            for key, item in options.items():
                prompt += f'{key}. {item}\n'
            prompt += 'Please select the correct answer from the options above.'

        msgs = self._image_messages(line, dataset)
        msgs.append(dict(type='text', value=prompt.rstrip()))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        import string

        import pandas as pd

        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}

        prompt = ''
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if options:
            prompt += 'Options:\n'
            for key, item in options.items():
                prompt += f'{key}. {item}\n'
            prompt += 'Answer with only the option letter from the given choices.'

        msgs = self._image_messages(line, dataset)
        msgs.append(dict(type='text', value=prompt.rstrip()))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        msgs = self._image_messages(line, dataset)
        msgs.append(dict(type='text', value=f"{line['question']} Please answer with only yes or no."))
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        msgs = self._image_messages(line, dataset)
        msgs.append(dict(type='text', value=line['question'] + '\nPlease answer concisely with short words or phrases when possible.'))
        return msgs


Qwen3_5VLPromptMixin = Qwen35VLPromptMixin

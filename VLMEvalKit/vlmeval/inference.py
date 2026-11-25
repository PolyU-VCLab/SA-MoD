import argparse
import os
import os.path as osp
import warnings

import torch
import torch.distributed as dist
from tqdm import tqdm

from vlmeval.config import supported_VLM
from vlmeval.smp import (dump, get_logger, get_pred_file_format, get_pred_file_path,
                         get_rank_and_world_size, load, upsert_dataset_status)
from vlmeval.utils import track_progress_rich

logger = get_logger(__name__)
FAIL_MSG = 'Failed to obtain answer via API.'
VISION_TOKEN_RETENTION_SUM_ATTR = '_vlmeval_vision_token_retention_sum'
VISION_TOKEN_RETENTION_NUM_SAMPLES_ATTR = '_vlmeval_vision_token_retention_num_samples'
VISION_TOKEN_RETENTION_VALUES_ATTR = '_vlmeval_vision_token_retention_values'
VISION_TOKEN_RETENTION_LAYER_SUMS_ATTR = '_vlmeval_vision_token_retention_layer_sums'
VISION_TOKEN_RETENTION_LAYER_COUNTS_ATTR = '_vlmeval_vision_token_retention_layer_counts'


def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Linear percentile. ``quantile`` is in ``[0, 1]``."""

    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _vision_token_retention_distribution(values: list[float]) -> dict:
    """Sample-level vision-token retention inside one benchmark. Values are in ``[0, 1]``."""

    ordered = sorted(float(value) for value in values)
    median = _percentile(ordered, 0.5)
    return {
        'min': ordered[0],
        'p25': _percentile(ordered, 0.25),
        'median': median,
        'p75': _percentile(ordered, 0.75),
        'mean': sum(ordered) / len(ordered),
        'max': ordered[-1],
        'num_samples': len(ordered),
    }


def _format_vision_token_retention_fields(
    overall_ratio: float,
    layer_ratios: dict[int, float],
    num_samples: int,
) -> dict:
    """Format vision-token retention for status.json (percentage strings).

    ``overall_ratio`` / ``layer_ratios`` values are in ``[0, 1]``.
    """
    fields = {
        'vision_token_retention(%)': f'{float(overall_ratio) * 100.0:.1f}(%)',
        'total_num_sample': int(num_samples),
    }
    if layer_ratios:
        max_idx = max(int(i) for i in layer_ratios.keys())
        parts = []
        for layer_idx in range(max_idx + 1):
            if layer_idx in layer_ratios:
                parts.append(f'{float(layer_ratios[layer_idx]) * 100.0:.1f}')
            else:
                parts.append('100.0')
        fields['vision_token_retention_per_layer'] = '|'.join(parts)
    else:
        fields['vision_token_retention_per_layer'] = ''
    return fields


def _model_runtime_metrics(model) -> dict:
    metrics = {}
    # Raw accumulators for multi-rank merge (ratios in [0, 1]).
    retention_num_samples = int(getattr(model, VISION_TOKEN_RETENTION_NUM_SAMPLES_ATTR, 0) or 0)
    if retention_num_samples > 0:
        retention_sum = float(getattr(model, VISION_TOKEN_RETENTION_SUM_ATTR, 0.0) or 0.0)
        metrics.update({
            'model_vision_token_retention_sum': retention_sum,
            'model_vision_token_retention_num_samples': retention_num_samples,
        })

    retention_values = getattr(model, VISION_TOKEN_RETENTION_VALUES_ATTR, None) or []
    if isinstance(retention_values, list) and retention_values:
        metrics['model_vision_token_retention_values'] = [float(value) for value in retention_values]

    layer_sums = getattr(model, VISION_TOKEN_RETENTION_LAYER_SUMS_ATTR, None) or {}
    layer_counts = getattr(model, VISION_TOKEN_RETENTION_LAYER_COUNTS_ATTR, None) or {}
    if isinstance(layer_sums, dict) and isinstance(layer_counts, dict):
        for layer_idx in sorted(layer_sums.keys()):
            count = int(layer_counts.get(layer_idx, 0) or 0)
            if count <= 0:
                continue
            layer_sum = float(layer_sums[layer_idx])
            metrics[f'model_vision_token_retention_layer_{int(layer_idx)}_sum'] = layer_sum
            metrics[f'model_vision_token_retention_layer_{int(layer_idx)}_num_samples'] = count
    return metrics


def _reset_model_vision_token_retention(model) -> None:
    """Start a fresh vision-token retention window for one benchmark."""
    scalar_defaults = {
        VISION_TOKEN_RETENTION_SUM_ATTR: 0.0,
        VISION_TOKEN_RETENTION_NUM_SAMPLES_ATTR: 0,
    }
    for attr, default in scalar_defaults.items():
        if hasattr(model, attr):
            setattr(model, attr, default)
    for attr in (VISION_TOKEN_RETENTION_LAYER_SUMS_ATTR, VISION_TOKEN_RETENTION_LAYER_COUNTS_ATTR):
        if hasattr(model, attr):
            setattr(model, attr, {})
    if hasattr(model, VISION_TOKEN_RETENTION_VALUES_ATTR):
        setattr(model, VISION_TOKEN_RETENTION_VALUES_ATTR, [])


def _merge_model_runtime_metrics(metric_items: list[dict]) -> dict:
    metric_items = [item for item in metric_items if isinstance(item, dict) and item]
    if not metric_items:
        return {}

    metrics = {}
    retention_num_samples = sum(
        int(item.get('model_vision_token_retention_num_samples', 0)) for item in metric_items
    )
    if retention_num_samples > 0:
        retention_sum = sum(float(item.get('model_vision_token_retention_sum', 0.0)) for item in metric_items)
        overall_ratio = retention_sum / retention_num_samples

        layer_ids = set()
        prefix = 'model_vision_token_retention_layer_'
        for item in metric_items:
            for key in item:
                if key.startswith(prefix) and key.endswith('_num_samples'):
                    mid = key[len(prefix):-len('_num_samples')]
                    if mid.isdigit():
                        layer_ids.add(int(mid))
        layer_rates = {}
        for layer_idx in sorted(layer_ids):
            layer_num = sum(
                int(item.get(f'{prefix}{layer_idx}_num_samples', 0)) for item in metric_items
            )
            if layer_num <= 0:
                continue
            layer_sum = sum(
                float(item.get(f'{prefix}{layer_idx}_sum', 0.0)) for item in metric_items
            )
            layer_rates[layer_idx] = layer_sum / layer_num

        metrics.update(
            _format_vision_token_retention_fields(overall_ratio, layer_rates, retention_num_samples)
        )
    retention_values = []
    for item in metric_items:
        raw_values = item.get('model_vision_token_retention_values') or []
        if isinstance(raw_values, list):
            retention_values.extend(float(value) for value in raw_values)
    if retention_values:
        metrics['vision_token_retention'] = _vision_token_retention_distribution(retention_values)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, retry_failed=True):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])
    # Build str→orig mapping for checkpoint key conversion
    index_str_to_orig = {str(i): i for i in indices}

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(dataset, 'force_use_dataset_prompt') and dataset.force_use_dataset_prompt:
            struct = dataset.build_prompt(item)
        elif hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_checkpoint.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('https://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data_v11 = load(v11_pred)
                ans_map = {str(x): y for x, y in zip(data_v11['index'], data_v11['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if retry_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}
        logger.info(f'Reuse {len(res)} inference results from previous run.')

    structs = [s for i, s in zip(indices, structs) if str(i) not in res]
    indices = [i for i in indices if str(i) not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        str_indices = [str(i) for i in indices]
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=str_indices)

    # Load the full accumulated results (str keys)
    if osp.exists(out_file):
        res = load(out_file)
    # Convert str keys back to original types for caller compatibility
    result = {index_str_to_orig[k]: v for k, v in res.items() if k in index_str_to_orig}
    if index_set is not None:
        result = {k: v for k, v in result.items() if k in index_set}
    return result


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False,
               retry_failed=True):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc,
            retry_failed=retry_failed)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(dataset, 'force_use_dataset_prompt') and dataset.force_use_dataset_prompt:
            struct = dataset.build_prompt(data.iloc[i])
        elif hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# Add for agent evaluation
def _is_structured_record(v):
    return isinstance(v, dict) and 'prediction' in v and 'extra_records' in v


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, retry_failed=True, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # The same model instance is reused across datasets. Retention counters live
    # on that instance, so reset them here to report this benchmark only.
    _reset_model_vision_token_retention(model)
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if retry_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)
    timing_tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}_model_runtime.pkl')

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm,
        retry_failed=retry_failed)
    dump(_model_runtime_metrics(model), timing_tmpl.format(rank))
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        runtime_metric_items = []
        for i in range(world_size):
            timing_file = timing_tmpl.format(i)
            if osp.exists(timing_file):
                runtime_metric_items.append(load(timing_file))
                os.remove(timing_file)
        runtime_metrics = _merge_model_runtime_metrics(runtime_metric_items)
        if runtime_metrics:
            logger.info(
                f'Model runtime metrics for {model_name}/{dataset_name}: {runtime_metrics}'
            )
            upsert_dataset_status(
                run_dir=work_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                runtime_metrics=runtime_metrics,
            )

        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        if os.getenv('SPLIT_THINK', False):
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                prediction = [data_all[x]['prediction'] for x in data['index']]
                extra_records = [data_all[x]['extra_records'] for x in data['index']]
                data['extra_records'] = extra_records
            else:
                prediction = [str(data_all[x]) for x in data['index']]

            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)
            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {split_func}')
            tups = [split_func(x) for x in prediction]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            # data['prediction'] = [str(data_all[x]) for x in data['index']]
            # Add for agent evaluation
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                data['prediction'] = [data_all[x]['prediction'] for x in data['index']]
                data['extra_records'] = [data_all[x]['extra_records'] for x in data['index']]
            else:
                data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
        # Clean up API checkpoint file
        checkpoint_file = f'{work_dir}/{model_name}_{dataset_name}_checkpoint.pkl'
        if osp.exists(checkpoint_file):
            os.remove(checkpoint_file)
        # Clean up PREV file
        if osp.exists(prev_file):
            os.remove(prev_file)
    if world_size > 1:
        dist.barrier()
    return model

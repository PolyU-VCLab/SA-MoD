# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""Unit tests for OHiSA PPO wiring: config routing and loss-branch exclusivity."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from verl.workers.utils.ohisa import (
    OhisaRuntimeCfg,
    build_response_token_mask,
    compute_hidden_trajectory_loss,
    get_text_model,
    project_token_mask_to_valid_span,
    resolve_ohisa_from_module,
    resolve_ohisa_from_text_config,
)


def test_resolve_ohisa_weight_zero_disabled():
    cfg = resolve_ohisa_from_text_config({"ohisa_loss_weight": 0.0})
    assert cfg.enabled is False
    # Key present → track/log even when λ==0.
    assert cfg.track is True
    assert cfg.mode == "on_only"


def test_resolve_ohisa_on_only():
    cfg = resolve_ohisa_from_text_config({"ohisa_loss_weight": 1.0})
    assert cfg.enabled is True
    assert cfg.mode == "on_only"


def test_resolve_from_module_reads_llm_config_when_text_config_has_no_weight():
    module = SimpleNamespace(config=SimpleNamespace(llm_config=SimpleNamespace(ohisa_loss_weight=1.0)))
    cfg = resolve_ohisa_from_module(module)
    assert cfg.enabled
    assert cfg.loss_weight == 1.0
    assert cfg.mode == "on_only"


def test_resolve_from_module_reads_text_config():
    module = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(ohisa_loss_weight=1.0)))
    cfg = resolve_ohisa_from_module(module)
    assert isinstance(cfg, OhisaRuntimeCfg)
    assert cfg.mode == "on_only"


def _make_toy_traj(batch: int = 2, seq: int = 4, hidden: int = 8, layers: int = 2):
    device = torch.device("cpu")
    layer_indices = list(range(layers))
    h_in = [torch.randn(batch, seq, hidden, device=device, requires_grad=True) for _ in layer_indices]
    h_out = [h + 0.1 for h in h_in]
    keep = [torch.ones(batch, seq, dtype=torch.bool, device=device) for _ in layer_indices]
    # Make one pad / dropped token so keep is non-trivial.
    keep[0][0, -1] = False
    traj = {
        "layer_indices": layer_indices,
        "hidden_states_in": h_in,
        "hidden_states_out": h_out,
        "keep_masks": keep,
        "position_embeddings": None,
        "text_position_ids": None,
        "attention_mask_2d": None,
        "causal_mask": None,
        "linear_attn_mask": None,
        "visual_pos_masks": None,
    }
    # HF tuple: index l is input to layer l.
    teacher_hiddens = tuple(torch.randn(batch, seq, hidden, device=device) for _ in range(layers + 1))
    return traj, teacher_hiddens


class _DummyTextModel(torch.nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(layer_types=None)
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n_layers)])


class _DummyRoot(torch.nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.model = SimpleNamespace(language_model=_DummyTextModel(n_layers))


def test_on_policy_runs_teacher_on_actor_hidden():
    traj, _teacher_hiddens = _make_toy_traj()
    actor = _DummyRoot()
    teacher = _DummyRoot()
    calls = {"teacher": []}

    def fake_teacher_forward(module, layer_idx, hidden, **kwargs):
        calls["teacher"].append(("full", layer_idx, id(hidden)))
        return hidden + 0.01

    with (
        mock.patch("verl.workers.utils.ohisa.teacher_forward_layer", side_effect=fake_teacher_forward),
        mock.patch("verl.workers.utils.ohisa.get_text_model", return_value=actor.model.language_model),
    ):
        loss, metrics = compute_hidden_trajectory_loss(
            traj,
            teacher,
        )

    assert torch.isfinite(loss)
    assert loss.requires_grad
    # Teacher transitions run on actor inputs.
    assert all(tag == "full" for tag, _, _ in calls["teacher"])
    assert len(calls["teacher"]) == len(traj["layer_indices"])
    assert metrics["ohisa/num_tokens"] > 0
    assert "naive_layerwise_distill_mse_loss" not in metrics


def test_on_only_is_response_only_mean_within_layer_then_across_layers():
    actor = _DummyRoot()
    teacher = _DummyRoot()
    h_in = [
        torch.zeros(1, 4, 1, requires_grad=True),
        torch.zeros(1, 4, 1, requires_grad=True),
    ]
    h_out = [
        torch.tensor([[[100.0], [100.0], [1.0], [1.0]]], requires_grad=True),
        torch.tensor([[[100.0], [100.0], [3.0], [100.0]]], requires_grad=True),
    ]
    traj = {
        "layer_indices": [0, 1],
        "hidden_states_in": h_in,
        "hidden_states_out": h_out,
        "keep_masks": [
            torch.tensor([[True, True, True, True]]),
            torch.tensor([[True, True, True, False]]),
        ],
        "position_embeddings": None,
        "text_position_ids": None,
        "attention_mask_2d": None,
        "causal_mask": None,
        "linear_attn_mask": None,
        "visual_pos_masks": None,
    }
    response_mask = torch.tensor([[False, False, True, True]])

    with (
        mock.patch(
            "verl.workers.utils.ohisa.teacher_forward_layer",
            side_effect=lambda module, layer_idx, hidden, **kwargs: torch.zeros_like(hidden),
        ),
        mock.patch(
            "verl.workers.utils.ohisa.get_text_model",
            return_value=actor.model.language_model,
        ),
    ):
        loss, metrics = compute_hidden_trajectory_loss(
            traj,
            teacher,
            response_token_mask=response_mask,
        )

    # Layer 0 response mean = 1; layer 1 has one kept response token, mean = 9.
    assert loss.item() == pytest.approx(5.0)
    assert metrics["ohisa/on_mse"] == pytest.approx(5.0)
    assert metrics["ohisa/num_tokens"] == 3.0
    assert metrics["ohisa/num_layers"] == 2.0


def test_response_token_mask_drops_prompt_positions():
    traj, _teacher_hiddens = _make_toy_traj(batch=2, seq=6, hidden=4, layers=1)
    actor = _DummyRoot(n_layers=1)
    teacher = _DummyRoot(n_layers=1)
    response_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)
    starts = torch.tensor([4, 4])
    resp = build_response_token_mask(6, response_mask, starts)
    assert resp[:, :4].sum() == 0
    assert int(resp.sum()) == 3

    def fake_teacher_forward(module, layer_idx, hidden, **kwargs):
        return hidden

    with (
        mock.patch("verl.workers.utils.ohisa.teacher_forward_layer", side_effect=fake_teacher_forward),
        mock.patch("verl.workers.utils.ohisa.get_text_model", return_value=actor.model.language_model),
    ):
        loss, metrics = compute_hidden_trajectory_loss(
            traj,
            teacher,
            response_token_mask=resp,
        )
    assert torch.isfinite(loss)
    assert metrics["ohisa/num_tokens"] == 3.0


def test_project_response_mask_follows_left_aligned_valid_tokens():
    # Left-padded collated rows. Valid tokens are packed left after rmpad.
    attention = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ]
    )
    response = build_response_token_mask(
        seq_len=5,
        response_mask=torch.tensor([[1, 1], [1, 0]]),
        response_start_idx=torch.tensor([3, 2]),
    )
    projected = project_token_mask_to_valid_span(response, attention)
    assert projected.shape == (2, 3)
    assert projected.tolist() == [[0, 1, 1], [0, 1, 0]]


def test_weight_zero_still_tracks_traj_for_logging():
    """λ==0: still request student trajectory so L_hidden can be logged."""
    cfg = resolve_ohisa_from_text_config({"ohisa_loss_weight": 0.0})
    student_kwargs = {}
    if cfg.track:
        student_kwargs["return_samod_trajectory"] = True
    assert student_kwargs == {"return_samod_trajectory": True}
    assert cfg.enabled is False
    assert cfg.mode == "on_only"


def test_enabled_on_distill_requests_expected_kwargs():
    cfg = resolve_ohisa_from_text_config({"ohisa_loss_weight": 1.0})
    student_kwargs = {}
    if cfg.track:
        student_kwargs["return_samod_trajectory"] = True
    assert student_kwargs == {"return_samod_trajectory": True}
    assert cfg.mode == "on_only"
    assert cfg.enabled is True


def test_get_text_model_unwraps_neo_causal_lm():
    layers = [object()]
    decoder = SimpleNamespace(layers=layers)
    causal_lm = SimpleNamespace(model=decoder)
    chat = SimpleNamespace(language_model=causal_lm)
    assert get_text_model(chat) is decoder

    qwen_text = SimpleNamespace(layers=layers)
    qwen = SimpleNamespace(model=SimpleNamespace(language_model=qwen_text))
    assert get_text_model(qwen) is qwen_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

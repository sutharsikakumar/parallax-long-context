"""Runnable reference stub for plugging a custom attention mechanism into lctx-bench.

This file is the thing to copy. It shows all three plug-in routes described in
``lctx.models.custom_attn``, with a deliberately simple mechanism standing in for
a real one so the wiring is visible without the kernel getting in the way.

The stand-in is **sliding-window attention**: softmax attention restricted to the
last ``window`` keys. It is a real mechanism (it changes what the model can
retrieve, which is exactly what a long-context benchmark should detect), it is
short enough to read, and swapping it for a linear / local-linear / covariance-
corrected mechanism or a fused Triton kernel means replacing one function body.

Run the self-check (no model download, no GPU)::

    python examples/custom_attention_stub.py

Then point a config at it::

    models:
      - name: my-model-swa
        adapter: custom_attn
        model_id: org/my-model
        tokenizer: {kind: hf}
        capture_attention: true          # optional, for probes
        params:
          register_hook: examples.custom_attention_stub:register
          attn_implementation: lctx_sliding_window
          # or, instead of the two lines above:
          # attention_patch: examples.custom_attention_stub:patch_model
          # patch_kwargs: {window: 512}

and run it exactly like any hosted model::

    lctx run -c configs/local_custom_attention.yaml
"""

from __future__ import annotations

import math
from typing import Any

WINDOW = 512


# ---------------------------------------------------------------------------
# Route 1: a registered attention function.
#
# `transformers` dispatches attention through a registry keyed by name, so a
# custom kernel can be published under a name and selected per-model with
# `attn_implementation`. This is the cleanest route for a kernel swap, because
# the model's module structure is untouched.
# ---------------------------------------------------------------------------

def sliding_window_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    window: int = WINDOW,
    **kwargs: Any,
):
    """Softmax attention restricted to the last ``window`` keys.

    Signature note: ``transformers`` calls attention functions with this shape and
    expects ``(attn_output, attn_weights)``. Return ``None`` for the weights if the
    mechanism never materialises them — lctx-bench handles that and simply reports
    that probes are unavailable.

    Replace the body with your mechanism. The only hard requirements are the
    returned shapes: ``attn_output`` is ``(batch, heads, queries, head_dim)`` and
    ``attn_weights``, when present, is ``(batch, heads, queries, keys)``.
    """
    import torch

    scaling = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])

    # Grouped-query attention: expand KV heads to match the query heads.
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[..., : key.shape[-2]]

    # Mask out everything further back than `window`.
    q_len, k_len = scores.shape[-2], scores.shape[-1]
    q_idx = torch.arange(k_len - q_len, k_len, device=scores.device).unsqueeze(-1)
    k_idx = torch.arange(k_len, device=scores.device).unsqueeze(0)
    too_old = (q_idx - k_idx) >= window
    scores = scores.masked_fill(too_old, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and getattr(module, "training", False):
        weights = torch.nn.functional.dropout(weights, p=dropout)

    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


def register() -> None:
    """Publish the mechanism under a name usable as ``attn_implementation``.

    lctx-bench calls this (via ``params.register_hook``) *before* the model is
    constructed, which is what makes the name resolvable at load time.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["lctx_sliding_window"] = sliding_window_attention


# ---------------------------------------------------------------------------
# Route 2: a patch function.
#
# Use this when the mechanism changes module structure rather than just the
# kernel — extra projections, learned state, a different cache layout.
# ---------------------------------------------------------------------------

def patch_model(model: Any, window: int = WINDOW, **kwargs: Any) -> Any:
    """Swap the attention behaviour of every decoder layer, in place.

    Must return the patched model. Raising here is better than silently patching
    nothing: a run that reports "custom attention" while measuring the stock model
    is worse than a run that fails.
    """
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise AttributeError(
            f"{type(model).__name__} has no `.model.layers`; adjust patch_model "
            "for this architecture"
        )

    patched = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        # Record the window on the module so the attention function can read it.
        attn.sliding_window = window
        attn.config._attn_implementation = "lctx_sliding_window"
        patched += 1

    if patched == 0:
        raise RuntimeError("patch_model matched no attention modules")
    model.config._attn_implementation = "lctx_sliding_window"
    register()
    return model


# ---------------------------------------------------------------------------
# Route 3: a complete custom loader.
#
# Use this when the model is not an `AutoModelForCausalLM` at all. The object
# returned only has to expose `.generate(...)`, `.config`, and `.device` the way
# HFAdapter uses them.
# ---------------------------------------------------------------------------

def load_my_model(model_id: str, checkpoint: str | None = None, **kwargs: Any) -> Any:
    """Build the model however you like, then return it ready for generation."""
    from transformers import AutoModelForCausalLM

    kwargs.pop("attn_implementation", None)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if checkpoint:
        import torch

        state = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[stub] {len(missing)} missing keys, e.g. {missing[:3]}")
        if unexpected:
            print(f"[stub] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return patch_model(model)


# ---------------------------------------------------------------------------
# Self-check: verifies the mechanism's shapes and semantics without downloading
# a model. Run this first when adapting the file — it catches the shape mistakes
# that otherwise surface as a silently wrong benchmark.
# ---------------------------------------------------------------------------

def _self_check() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed; install 'lctx-bench[hf]' to run this check")
        return 0

    batch, heads, q_len, head_dim, window = 1, 4, 64, 16, 8
    torch.manual_seed(0)
    q = torch.randn(batch, heads, q_len, head_dim)
    k = torch.randn(batch, heads, q_len, head_dim)
    v = torch.randn(batch, heads, q_len, head_dim)

    causal = torch.full((q_len, q_len), torch.finfo(torch.float32).min).triu(1)
    out, weights = sliding_window_attention(
        None, q, k, v, attention_mask=causal, window=window
    )

    assert out.shape == (batch, q_len, heads, head_dim), f"bad output shape {out.shape}"
    assert weights.shape == (batch, heads, q_len, q_len), f"bad weight shape {weights.shape}"

    rows = weights[0, 0]
    assert torch.allclose(rows.sum(-1), torch.ones(q_len), atol=1e-5), "rows must sum to 1"
    # Causality: no attention to the future.
    assert rows.triu(1).abs().max() < 1e-6, "attended to future positions"
    # The window is actually enforced.
    assert rows[q_len - 1, : q_len - window].abs().max() < 1e-6, "window not enforced"
    # And it is genuinely different from full attention.
    _, full = sliding_window_attention(
        None, q, k, v, attention_mask=causal, window=q_len
    )
    assert not torch.allclose(weights, full), "window had no effect"

    print("self-check passed:")
    print(f"  output  {tuple(out.shape)}")
    print(f"  weights {tuple(weights.shape)}, rows sum to 1, causal, window={window} enforced")
    print("\nnow point a config at this file — see the module docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())

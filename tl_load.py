"""Load the model trained by tf.py into TransformerLens.

Usage (in a notebook or script, same conda env):

    from tl_load import load_tl_model, encode, decode
    model = load_tl_model()
    logits, cache = model.run_with_cache(torch.tensor([encode("some text")]))

Three things in tf.py are non-standard, and this script accounts for them:

1. Attention scaling: tf.py divides scores by sqrt(n_embd)=sqrt(384), not
   sqrt(head_size)=sqrt(192) (in Head.forward, C is x.shape[-1] = n_embd).
   TransformerLens defaults to sqrt(d_head), so we set attn_scale explicitly.

2. No LayerNorm (is_layer_norm=False), so normalization_type=None.

3. The document mask (attention blocked across '^' separators) has no
   TransformerLens equivalent. On text that contains no '^', tf.py's mask
   reduces exactly to plain causal masking, so the two models agree.
   => Only analyze prompts WITHOUT '^' in them, or the logits will differ.
"""

import math

import pandas as pd
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Must match tf.py exactly
block_size = 512
n_embd = 384
n_layer = 1
n_head = 2
head_size = n_embd // n_head
train_path = "train.csv"
model_path = "weights.pth"

# Rebuild the exact vocab tf.py built (it derives it from train.csv at runtime,
# so any change to train.csv silently changes the token ids).
_dataset = "^".join(pd.read_csv(train_path, nrows=100000).dropna()["text"])
vocab = sorted(set(_dataset))
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)


def load_tl_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = HookedTransformerConfig(
        n_layers=n_layer,
        d_model=n_embd,
        n_ctx=block_size,
        d_head=head_size,
        n_heads=n_head,
        d_mlp=4 * n_embd,
        d_vocab=len(vocab),
        act_fn="relu",
        normalization_type=None,
        attn_scale=math.sqrt(n_embd),  # tf.py quirk #1
        positional_embedding_type="standard",
    )

    sd = torch.load(model_path, map_location="cpu")

    # nn.Linear stores weight as [out, in]; TransformerLens stores [in, out],
    # hence all the .T below.
    new_sd = {
        "embed.W_E": sd["token_embedding_table.weight"],
        "pos_embed.W_pos": sd["position_embedding_table.weight"],
        "unembed.W_U": sd["lm_head.weight"].T,
        "unembed.b_U": sd["lm_head.bias"],
    }
    for l in range(n_layer):
        p = f"blocks.{l}"
        for name, tl_name in [("query", "W_Q"), ("key", "W_K"), ("value", "W_V")]:
            new_sd[f"{p}.attn.{tl_name}"] = torch.stack(
                [sd[f"{p}.sa.heads.{h}.{name}.weight"].T for h in range(n_head)]
            )
        # proj consumes the head outputs concatenated head0|head1, which is the
        # same as TL's per-head W_O summed over heads after this reshape.
        new_sd[f"{p}.attn.W_O"] = (
            sd[f"{p}.sa.proj.weight"].T.reshape(n_head, head_size, n_embd)
        )
        new_sd[f"{p}.attn.b_O"] = sd[f"{p}.sa.proj.bias"]
        for b in ["b_Q", "b_K", "b_V"]:
            new_sd[f"{p}.attn.{b}"] = torch.zeros(n_head, head_size)
        new_sd[f"{p}.mlp.W_in"] = sd[f"{p}.ff.network.0.weight"].T
        new_sd[f"{p}.mlp.b_in"] = sd[f"{p}.ff.network.0.bias"]
        new_sd[f"{p}.mlp.W_out"] = sd[f"{p}.ff.network.2.weight"].T
        new_sd[f"{p}.mlp.b_out"] = sd[f"{p}.ff.network.2.bias"]

    model = HookedTransformer(cfg)
    # strict=False: TL keeps extra buffers (causal mask etc.) not in our dict
    model.load_state_dict(new_sd, strict=False)
    return model.to(device)


@torch.no_grad()
def verify(original_model, tl_model, text="the quick brown fox"):
    """Check both models produce the same logits on a '^'-free prompt.

    Call from gpt_testbed.ipynb where the tf.py model instance exists:
        verify(m, load_tl_model())
    """
    assert "^" not in text
    tokens = torch.tensor([encode(text)], device=next(tl_model.parameters()).device)
    orig_logits, _ = original_model(tokens)
    tl_logits = tl_model(tokens)
    max_diff = (orig_logits - tl_logits).abs().max().item()
    print(f"max logit diff: {max_diff:.2e}")
    assert max_diff < 1e-3, "conversion mismatch — do not trust the TL model"
    return max_diff


if __name__ == "__main__":
    model = load_tl_model()
    text = "the quick brown fox"
    tokens = torch.tensor([encode(text)], device=next(model.parameters()).device)
    logits, cache = model.run_with_cache(tokens)
    print("loaded OK; cached activations:")
    for k in cache.keys():
        print(" ", k, tuple(cache[k].shape))

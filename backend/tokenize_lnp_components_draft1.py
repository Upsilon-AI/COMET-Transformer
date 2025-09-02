# pip install transformers numpy   # (embeddings require: pip install torch)
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer

def tokenize_lnp_components(
    lnp_dataframe: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 64,
    component_cols: Optional[List[str]] = None,
    return_embeddings: bool = False,   # set True only if torch is installed
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Turn LNP component columns into transformer-ready token IDs.
    Optionally also returns mean-pooled embeddings.

    Returns a dict keyed by component name:
      {
        "<component>": {
           "input_ids": [N, L] np.int64,
           "attention_mask": [N, L] np.int64,
           "texts": list[str],
           (optional) "embeddings": [N, D] np.float32
        },
        ...
      }
    """
    df = lnp_dataframe.copy()

    # 1) pick component columns
    if component_cols is None:
        patterns = [r"\bioniz", r"\bchol", r"\bdspc\b", r"\bpeg\b"]
        component_cols = [
            c for c in df.columns
            if any(re.search(p, c.lower()) for p in patterns)
        ]
    if not component_cols:
        raise ValueError("No component columns found. Pass component_cols=[...] explicitly.")

    # 2) tiny value formatter
    def fmt(v) -> str:
        if pd.isna(v): return "unknown"
        try:
            x = float(v)
            return f"{x:.4g}"
        except Exception:
            return str(v)

    # 3) tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 4) build per-component texts and tokenize
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for comp in component_cols:
        texts = [f"component {comp}: {fmt(v)}" for v in df[comp]]
        enc = tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        pack = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "texts": texts,
        }
        out[comp] = pack

    # 5) (optional) compute embeddings via mean pooling
    if return_embeddings:
        try:
            import torch
            from transformers import AutoModel
        except Exception:
            raise RuntimeError(
                "Embeddings require PyTorch. Install with `pip install torch` or set return_embeddings=False."
            )
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        with torch.no_grad():
            for comp, pack in out.items():
                ids  = torch.tensor(pack["input_ids"])
                mask = torch.tensor(pack["attention_mask"])
                out_h = model(input_ids=ids, attention_mask=mask).last_hidden_state  # [N,L,D]
                # masked mean pool
                mask_f = mask.unsqueeze(-1)  # [N,L,1]
                summed = (out_h * mask_f).sum(dim=1)              # [N,D]
                counts = mask_f.sum(dim=1).clamp(min=1)           # [N,1]
                emb = (summed / counts).cpu().numpy().astype("float32")
                pack["embeddings"] = emb

    return out

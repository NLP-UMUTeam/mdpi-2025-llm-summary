from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import random
import numpy as np

from transformers import AutoTokenizer, AutoModel, pipeline
import torch

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

class HFNER:
    def __init__(self, model_name="MMG/xlm-roberta-large-ner-spanish", device: int = -1):
        self.ner = pipeline("token-classification", model=model_name, aggregation_strategy="simple", device=device)

    def extract(self, text: str) -> Dict[str,List[str]]:
        ents = self.ner(text)
        out: Dict[str,List[str]] = {}
        for e in ents:
            label = e["entity_group"]
            word  = e["word"]
            out.setdefault(label, []).append(word)
        return out
        
class AttnScorer:
    def __init__(self, model_name="dccuchile/bert-base-spanish-wwm-cased", device: Optional[str]=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
        self.device = device
        if device:
            self.model.to(device)

    @torch.no_grad()
    def score(self, text: str, agg: str = "max", max_length: int = 1024) -> float:
        if not text.strip():
            return 0.0
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        if self.device:
            enc = {k: v.to(self.device) for k,v in enc.items()}
        out = self.model(**enc)
        atts = out.attentions  # list[L] (B,H,T,T)
        att_mean = torch.stack(atts).mean(0)   # (B,H,T,T)
        att_mean = att_mean.mean(1)[0]         # (T,T)
        to_cls = att_mean[:,0].detach().cpu().numpy()  # atención hacia [CLS]
        if len(to_cls) <= 1:
            raw = 0.0
        else:
            vals = to_cls[1:]
            mean_v = float(np.mean(vals))
            max_v  = float(np.max(vals))
            raw = mean_v if agg=="mean" else max_v if agg=="max" else 0.5*(mean_v+max_v)
        # normalizar con sigmoide centrado en 0.5
        return float(1 / (1 + math.exp(-4*(raw-0.5))))
    

def compute_probs_with_entities(
    texts: List[str],
    attn_scores: List[float],
    ner: HFNER,
    w0: float = -0.6,
    w_attn: float = 2.0,
    w_len: float = 0.2,
    w_sent: float = 0.3,
    w_ent_count: float = 0.35,
    w_ent_TYPES: Dict[str, float] = None
) -> List[Dict]:
    if w_ent_TYPES is None:
        w_ent_TYPES = {"PER": 0.25, "ORG": 0.35, "LOC": 0.2, "MISC": 0.15}

    items = []
    for txt, a in zip(texts, attn_scores):
        len_norm = clamp01(len(txt)/200.0)
        is_sentence = 1.0 if txt.strip().endswith(('.', '!', '?')) else 0.0

        ents = ner.extract(txt)  # ahora lista de strings
        ent_total = sum(len(v) for v in ents.values())
        ent_count_norm = clamp01(ent_total/3.0)

        z = w0 + w_attn*a + w_len*len_norm + w_sent*is_sentence + w_ent_count*ent_count_norm
        for t, vs in ents.items():
            z += w_ent_TYPES.get(t, 0.1) * clamp01(len(vs)/2.0)

        items.append({
            "text": txt,
            "attn": round(a,4),
            "ents": ents,
            "prob": sigmoid(z)
        })
    return items


def select_bottleneck(items: List[Dict], budget=6, mode="topk", seed=42) -> List[Dict]:
    rng = random.Random(seed)
    items = sorted(items, key=lambda x:x["prob"], reverse=True)
    if mode=="topk":
        return items[:budget]
    selected=[]
    for it in items:
        if len(selected)>=budget: break
        if rng.random() <= it["prob"]:
            selected.append(it)
    if len(selected)<min(budget,len(items)):
        for it in items:
            if it not in selected:
                selected.append(it)
            if len(selected)>=budget: break
    return selected

def weight_tag(p: float, scheme="discrete") -> str:
    if scheme=="discrete":
        if p>=0.65: return "<<w=H>>"
        if p>=0.45: return "<<w=M>>"
        return "<<w=L>>"
    return f"<<w={int(round(100*p))}>>"

def ents_bracket(ents: Dict[str,int]) -> str:
    if not ents: return "[ENTS none]"
    parts = [f"{k}:{v}" for k,v in sorted(ents.items())]
    return "[ENTS " + ", ".join(parts) + "]"

def build_highlight_prompt(highlights):
    scorer = AttnScorer(device="cuda:0")
    attn_scores=[scorer.score(h) for h in highlights]
    # print(attn_scores)
    ner = HFNER("MMG/xlm-roberta-large-ner-spanish", device="cuda:0")
    items = compute_probs_with_entities(highlights, attn_scores, ner)  
    # print(items) 
    selected = select_bottleneck(items, budget=6, mode="topk")
    # print(selected)
    
    lines=[]
    for it in selected:
        tag = weight_tag(it["prob"], "discrete")
        ent_blk = ents_bracket(it["ents"])
        lines.append(f"{tag} {ent_blk} {it['text'].strip()}")
    
    print("\n".join(lines))
    return "\n".join(lines)

# if __name__ == "__main__":
#     highlights=[
#         "BioNova adquirió GenX por 450 millones de euros.",
#         "La operación se financiará principalmente con deuda.",
#         "El cierre se prevé para el cuarto trimestre.",
#         "La integración será gradual.",
#         "Analistas anticipan sinergias en I+D y manufactura.",
#         "El CEO no ofreció detalles técnicos."
#     ]

#     build_highlight_prompt(highlights)
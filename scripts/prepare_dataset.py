import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from ast import literal_eval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

# Parámetros de rendimiento/calidad
BATCH_SIZE = 32              # súbelo si tienes más VRAM
MAX_INPUT_LEN = 512
MAX_SUMMARY_LEN = 84         # 84/86 según tu uso
NUM_BEAMS = 4                # 1 = sin beam search (más rápido). Usa 2–4 si quieres algo mejor.
NO_REPEAT_NGRAM = 2

RE_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-Ü])')
WHITESPACE_HANDLER = lambda k: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', k.strip()))

@torch.inference_mode()
def load_model(model_name: str = MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if DEVICE.type == "cuda":
        model = model.half()  # FP16 en GPU
    model = model.to(DEVICE)
    model.eval()
    return tok, model

def split_text(article: str, window_size: int = 5) -> list:
    """Divide el artículo en bloques de ~window_size oraciones."""
    if not isinstance(article, str) or not article.strip():
        return []
    phases = RE_SENT_SPLIT.split(article)
    phases = [p for p in phases if p.strip()]
    chunks, temp = [], []
    for ph in phases:
        temp.append(ph)
        if len(temp) >= window_size:
            chunks.append(" ".join(temp))
            temp = []
    if temp:  # resto
        chunks.append(" ".join(temp))
    return chunks

def summarize_batch(texts, tokenizer, model):
    """Resume una lista de textos en un solo batch."""
    if not texts:
        return []
    # Preprocesamiento ligero
    texts = [WHITESPACE_HANDLER(t) for t in texts]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,                 # "longest" por defecto -> más rápido que pad a 512 fijo
        truncation=True,
        max_length=MAX_INPUT_LEN
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=MAX_SUMMARY_LEN,
        no_repeat_ngram_size=NO_REPEAT_NGRAM,
        num_beams=NUM_BEAMS,
    )

    outs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outs

def summarize_paragraph_list(paragraphs, tokenizer, model):
    """Versión batcheada para una lista de párrafos."""
    summaries = []
    for i in range(0, len(paragraphs), BATCH_SIZE):
        batch = paragraphs[i:i+BATCH_SIZE]
        summaries.extend(summarize_batch(batch, tokenizer, model))
    return summaries

def main():
    # Carga modelo una sola vez
    # tokenizer, model = load_model(MODEL_NAME)
    # 
    # # Lee dataset y muestrea
    # df = pd.read_csv("./fine-tuning_dataset/dataset_dacsa_test.csv")
    # df = df.dropna(subset=["id", "summary", "article"])  # conserva campos clave
    # df = df.sample(n=1000, random_state=1).reset_index(drop=True)
    # 
    # # Prepara columnas de salida
    # df["highlights"] = None  # guardaremos lista de strings (JSON al exportar)
    # 
    # # Procesa fila a fila pero resumiendo por lotes dentro de cada artículo
    # for idx in tqdm(range(len(df))):
    #     article = df.at[idx, "article"]
    #     paragraphs = split_text(article)  # ~5 oraciones por chunk
    #     if not paragraphs:
    #         df.at[idx, "highlights"] = []
    #         continue
    #     highlights = summarize_paragraph_list(paragraphs, tokenizer, model)
    #     df.at[idx, "highlights"] = highlights
    # 
    # # Exporta una sola vez
    # df.to_csv("./fine-tuning_dataset/dataset_dacsa_test_highlight.csv", index=False)

    df = pd.read_csv("./fine-tuning_dataset/dataset_dacsa_test_highlight.csv")
    df["highlights"] = df["highlights"].apply(lambda x: " ".join(literal_eval(x)))
    df.to_csv("./results/highlights_results.csv", index=False)
    
if __name__ == "__main__":
    main()

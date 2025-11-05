# evaluator.py
import argparse
import ast
import evaluate
from bert_score import BERTScorer
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_metrics_batch(refs, preds, rouge_metric, bert_scorer):
    # ROUGE (agregamos por lote sin promediar internamente)
    rouge_scores = rouge_metric.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True,
        use_aggregator=False
    )
    # rouge devuelve dict con listas por métrica
    # Aseguramos claves esperadas
    for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if k not in rouge_scores:
            # Si falta rougeLsum (por ejemplo), creamos zeros
            rouge_scores[k] = [0.0] * len(preds)

    # BERTScore (devolvemos F1 por ejemplo)
    P, R, F1 = bert_scorer.score(preds, refs)
    bert_f1 = F1.tolist()

    # Longitudes y compresión
    pred_len = [len(p.split()) for p in preds]
    ref_len  = [len(r.split()) for r in refs]
    compression = [ (pl / rl) if rl > 0 else 0.0 for pl, rl in zip(pred_len, ref_len) ]

    rows = []
    for i in range(len(preds)):
        rows.append({
            "rouge1": float(rouge_scores["rouge1"][i]) * 100,
            "rouge2": float(rouge_scores["rouge2"][i]) * 100,
            "rougeL": float(rouge_scores["rougeL"][i]) * 100,
            "rougeLsum": float(rouge_scores["rougeLsum"][i]) * 100,
            "bert_f1": float(bert_f1[i]) * 100 ,
            "mean_rougeL_bert": float(np.mean([rouge_scores["rougeL"][i], bert_f1[i]]))  * 100,
            "pred_len": int(pred_len[i]) * 100,
            "ref_len": int(ref_len[i]) * 100,
            "compression": float(compression[i]) * 100,
            "pred_empty": int(len(preds[i].strip()) == 0) * 100,
            "ref_empty": int(len(refs[i].strip()) == 0) * 100,
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True,
                        help="CSV con columnas: description (gold), description_llm (pred)")
    parser.add_argument("--pred_col", default="description_llm",
                        help="Nombre de la columna con la predicción del modelo")
    parser.add_argument("--ref_col", default="summary",
                        help="Nombre de la columna con el resumen gold")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_csv", default="scores.csv")
    args = parser.parse_args()

    # Cargar métricas/objetos globales
    rouge = evaluate.load("rouge")
    bert_scorer = BERTScorer(
        model_type="bert-base-uncased",
        # model_type="dccuchile/bert-base-spanish-wwm-cased",
        # num_layers=12,
        lang="es",
        rescale_with_baseline=False # hace comparables entre corridas
    )

    df = pd.read_csv(args.input_csv)
    # Limpieza básica
    df = df.dropna(subset=[args.ref_col, args.pred_col]).copy()
    df[args.ref_col] = df[args.ref_col].astype(str)
    df[args.pred_col] = df[args.pred_col].astype(str)

    all_rows = []
    # batching
    for i in tqdm(range(0, len(df), args.batch_size)):
        batch = df.iloc[i:i+args.batch_size]
        refs = batch[args.ref_col].tolist()
        preds = batch[args.pred_col].tolist()
        rows = compute_metrics_batch(refs, preds, rouge, bert_scorer)
        all_rows.extend(rows)

    # Añadimos como columnas planas
    scores_df = pd.DataFrame(all_rows, index=df.index)
    out = pd.concat([df, scores_df], axis=1)

    # Estadísticas rápidas en consola
    agg = out[["rouge1","rouge2","rougeL","rougeLsum","bert_f1","mean_rougeL_bert","compression"]].mean()
    print("\n=== Averages ===")
    print(agg.to_string())

    out.to_csv(args.output_csv, index=False)
    print(f"\nGuardado en: {args.output_csv}")

if __name__ == "__main__":
    main()

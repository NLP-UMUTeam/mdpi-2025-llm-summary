from transformers import pipeline
import torch 
import pandas as pd 
from tqdm import tqdm 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # summarizer = pipeline("summarization", model="ELiRF/mbart-large-cc25-dacsa-es", device=device)
    summarizer = pipeline("summarization", model="ELiRF/mt5-base-dacsa-es", device=device)
    
    df = pd.read_csv("./fine-tuning_dataset/dataset_dacsa_test.csv")
    df = df.sample(n=1000, random_state=1)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
    
        out = summarizer(row["article"], truncation=True)
        print(out)
        df.at[idx, "description_llm"] = out[0]["summary_text"]
        df.to_csv("./results/mt5_dacsa.csv", index=False)



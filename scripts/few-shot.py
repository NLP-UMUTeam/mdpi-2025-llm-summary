from abc import ABC, abstractmethod
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import sys
import gc
import argparse
from transformers import pipeline
import time
import gc
import LlmBase 
import os
from tqdm import tqdm 
import argparse
import pandas as pd 
import json
import LlmBase
import random

article_prompt = """
Summarize the following news article in one short paragraph (no more than 6 lines) in Spanish. Use clear, direct, and engaging language in Spanish. Make sure to include what happened, who is involved, when it happened, and why it is important.  Return only the final description — do not include explanations, titles, or any extra formatting.

Examples: 
{examples}

News Article:
{article}
"""

ner_bottleneck_prompt = """
Summarize the following news article in one short paragraph (no more than 6 lines) in Spanish. 
Use clear, direct, and engaging language in Spanish. 
Make sure to include what happened, who is involved, when it happened, and why it is important.  

Use the provided highlights and named entities as the main basis for your summary:
- Prioritize information from highlights tagged as <<w=H>> (high importance), include <<w=M>> if they add context, and avoid <<w=L>> unless needed for coherence.
- Ensure that all listed entities (people, organizations, locations, dates) appear in the summary consistently and correctly.
- If there are multiple entities of the same type, select the most relevant ones.

Return only the final description — do not include explanations, titles, or any extra formatting.

Examples: 
{examples}

News Article:
{article}

Highlights and Entities:
{highlights_with_entities}

"""

examples_prompt = """

News Article: 
{article}

Summary: 
{summary}
"""

class FewShot(LlmBase.AbstractLLM):
    name = 'few-shot'

    def __init__(self, model_name, quantization_config=None, temperature=0.5, top_k=50, top_p=0.9):
        super().__init__(model_name, quantization_config, temperature, top_k, top_p)
        self.system_input = """You are a professional digital news editor. Your task is to generate short, clear, and engaging descriptions of news articles for use in social media, newsletters, or headline summaries. Your writing style is concise, informative, and compelling. Always include the key elements: what happened, who is involved, when it happened, and why it matters. Maintain a neutral, journalistic tone while keeping the reader's attention with sharp, relevant summaries.  Return only the final description — do not include explanations, titles, or any extra formatting."""
        self.generated_tokens = 0
        self.total_time = 0
        self.start_time = 0  # Store the start time for real-time calculations
        self.response = ""
    
    def set_prompt(self, article, examples):
        initial_prompt = article_prompt.format(article=article, examples=examples)

        conversation = []

        if self.model_name not in {"gemma-2-2b", "gemma-2-9b", "gemma-2-27b"}:
            conversation.append({"role": "system", "content": self.system_input})
            
        conversation.append({"role": "user", "content": initial_prompt}) 
        prompt = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
        )
        self.prompt = prompt 
    

    def run_inference(self):
        self.response = ""
        self.generated_tokens = 0
        self.start_time = time.time()

        inputs = self.tokenizer([self.prompt], return_tensors='pt', add_special_tokens=False).to("cuda")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=4096,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_beams=1,
        )

        try:
            with torch.no_grad():
                self.model.generate(**generate_kwargs)

            print("\n------------------------------------")
            print("\nResponse:")

            for text in streamer:
                self.response += text
                tokens = self.tokenizer.encode(text)
                non_empty_tokens = [token for token in tokens if token != self.tokenizer.pad_token_id and token != self.tokenizer.eos_token_id]
                self.generated_tokens += len(non_empty_tokens)

                sys.stdout.write(text)
                sys.stdout.flush()

            end_time = time.time()
            elapsed = end_time - self.start_time
            tokens_per_second = self.generated_tokens / elapsed if elapsed > 0 else 0
            self.total_time += elapsed
            overall_tokens_per_second = self.generated_tokens / self.total_time if self.total_time > 0 else 0

            print(f"\n\nTotal tokens generated: {self.generated_tokens}")
            print(f"Tokens per second (current file): {tokens_per_second:.2f}")
            print(f"Total tokens per second (all files): {overall_tokens_per_second:.2f}")

            return self.generated_tokens, tokens_per_second, overall_tokens_per_second

        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] Skipping article due to CUDA OOM: {e}")
            self.response = "[ERROR: CUDA Out of Memory]"
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            return 0, 0, 0
         
    def get_response(self):
        return self.response


def save_statistics_to_csv(file_name, statistics):
    # Convert statistics to a pandas DataFrame
    df = pd.DataFrame(statistics, columns=['Model', 'File', 'Total Tokens', 'Tokens per Second (Current)', 'Tokens per Second (Total)'])
    
    # Append the statistics to the CSV file (write header only if the file doesn't exist)
    df.to_csv(file_name, header=not os.path.exists(file_name), index=False)
    
    
def generate_dataset(): 
    news_dir = "/home/ronghao/ACL-alignment/news_v2_2025/"
    list_dir = os.listdir(news_dir)
    
    for dir in list_dir: 
        files_dir = os.path.join(news_dir, dir)
        list_files = os.listdir(files_dir)
        data = []
        for file in list_files: 
            file_path = os.path.join(files_dir, file)
            with open(file_path, 'r') as file:
                content = json.load(file)
                file_id = os.path.splitext(file_path)[0]
                if 'description' in content and 'web_text' in content:
                    data.append({
                        'id': file_id,
                        'description': content['description'],
                        'web_text': content['web_text']
                    })
        
        json_df = pd.DataFrame(data)
    
    json_df.to_csv("dataset.csv", index=False)
    
    
def main():
    parser = argparse.ArgumentParser(description="Run Few-Shot NER inference.")
    parser.add_argument('--model', type=str, required=True, help="Model name to use")
    parser.add_argument('--type', default=None, help="Model to use")
    parser.add_argument('--folder', type=str)
    parser.add_argument('--shot', type=int)
    args = parser.parse_args()
    
    models = [
            "llama-3.1-8b", "llama-3.1-70b", 'gemma-2-2b', 
            'gemma-2-9b',"gemma-2-27b", 'mixtral', 'mistral-7b',
            'phi-3.5', 'phi-3.5-moe', 'qwen2.5-7b', 'qwen2.5-1.5b'
    ]
    
    
    model_name = args.model
    quantization = args.type
    folder = args.folder
    
    df = pd.read_csv("./fine-tuning_dataset/dataset_dacsa_test.csv")
    df = df.sample(n=1000, random_state=1)

    train_df = pd.read_csv("./fine-tuning_dataset/dataset_dacsa_train.csv")
    train_df = train_df.sample(n=args.shot, random_state=1)
    examples = ""
    
    for idx, row in train_df.iterrows():
        shot_examples = examples_prompt.format(article=row["article"], summary=row["summary"])
        examples += shot_examples 
    
    model = FewShot(model_name, quantization) 
    
    df["description_llm"] = ""

    for idx, row in df.iterrows(): 
        article = row["article"]
        model.set_prompt(article, examples)
        model.run_inference()
        response = model.get_response()
        if "[ERROR" in response:
            print(f"Skipping due to error in article {idx}")
        else:
            df.at[idx, "description_llm"] = response
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
            
        df.to_csv(f"{folder}/DACSA_test_few_shot_{args.shot}_{model_name}_{quantization}_results.csv", index=False)

    
if __name__ == '__main__':
    main() 
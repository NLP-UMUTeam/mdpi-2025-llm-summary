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

torch.set_float32_matmul_precision('high')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bnb_config_4bits = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

bnb_config_8bits = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def read_txt_file_as_string (file_path): 
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    return file_content


def clean_gpu_cache():
    """Force garbage collection and clear CUDA memory after each inference"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class AbstractLLM(ABC):
    _model_cache = {}

    def __init__(self, model_name, quantization_config, temperature, top_k, top_p):
        if model_name not in {
            "llama-3.1-8b", "llama-3.1-70b", "llama-3.2-1b", "llama-3.2-3b",
            "llama-2-7b", "llama-2-13b", "llama-2-70b", 'gemma-2-2b', 
            'gemma-2-9b', 'mixtral', 'mistral-7b', 'salamandra-2b', 'salamandra-7b',
            'gemma-2-27b', 'mixtral-8x22b', 'phi-3.5', 'phi-3.5-moe', 'qwen2.5-7b', 
            'qwen2.5-1.5b', 'hymba-1.5b', 'deepseek-qwen-32b', 'deepseek-llama-70b'}:
            raise Exception('Invalid model')

        self.model_name = model_name
        self.prompt = "" 
        self.response = ""

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Set quantization config if specified
        if quantization_config == "4bits": 
            self.quantization_config = bnb_config_4bits
        elif quantization_config == "8bits": 
            self.quantization_config = bnb_config_8bits
        else:
            self.quantization_config = None  # No quantization

        self.device = device  # Assuming `device` is defined globally or passed in some way

        # Check the model cache before loading
        if model_name in AbstractLLM._model_cache:
            self.model, self.tokenizer, self.terminators = AbstractLLM._model_cache[model_name]
        else:
            # Load model and tokenizer based on the model name
            self._load_model_and_tokenizer()

            # Cache the model for future use
            AbstractLLM._model_cache[model_name] = (self.model, self.tokenizer, self.terminators)

    def _load_model_and_tokenizer(self):
        """Helper function to load the model and tokenizer based on model_name."""
        # Mapping of model names to model IDs
        model_id_map = {
            "llama-3.1-8b": "/data/ronghao/LLM/meta-llama/Llama-3.1-8B-Instruct",
            "llama-3.1-70b": "/data/ronghao/LLM/meta-llama/Llama-3.1-70B-Instruct",
            "gemma-2-2b": "/data/ronghao/LLM/google/gemma-2-2b-it",
            "gemma-2-9b": "/data/ronghao/LLM/google/gemma-2-9b-it",
            "gemma-2-27b": "/data/ronghao/LLM/google/gemma-2-27b-it",
            "mixtral": "/data/ronghao/LLM/mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistral-7b": "/data/ronghao/LLM/mistralai/Mistral-7B-Instruct-v0.3", 
            "phi-3.5": "/data/ronghao/LLM/microsoft/Phi-3.5-mini-instruct",
            "phi-3.5-moe": "/data/ronghao/LLM/microsoft/Phi-3.5-MoE-instruct", 
            "qwen2.5-7b": "/data/ronghao/LLM/Qwen/Qwen2.5-7B-Instruct",
            "deepseek-qwen-32b": "/data/ronghao/LLM/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
            "deepseek-llama-70b": "/data/ronghao/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 
            "qwen2.5-1.5b": "/data/ronghao/LLM/Qwen/Qwen2.5-1.5B-Instruct",             
        }

        self.model_id = model_id_map[self.model_name]

        # Load model with or without quantization config based on its value
        if self.quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, use_cache=False, quantization_config=self.quantization_config, device_map='auto')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, use_cache=False, device_map='auto')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.terminators = [self.tokenizer.eos_token_id]

        # If specific models have different terminators, adjust as needed
        if self.model_name in {"llama-3.1-8b", "llama-3.1-70b", "llama-3.2-1b", "llama-3.2-3b"}:
            self.terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    def get_model(self):
        return self.model 

    @abstractmethod
    def set_prompt(self):
        pass 

    @abstractmethod
    def run_inference(self):
        pass

    @abstractmethod
    def get_response(self): 
        pass 

    def save_response(self, folder, name): 
        with open(f"./experiments/{self.model_name}/{folder}/{name}.txt", "w") as text_file:
            text_file.write(self.response)


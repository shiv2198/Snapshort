#importing libraries
import numpy as np 
import pandas as pd 
import re
import torch
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModelWithLMHead

def loadComponent():
    tokenizer = AutoTokenizer.from_pretrained("deep-learning-analytics/wikihow-t5-small")
    model = AutoModelWithLMHead.from_pretrained("deep-learning-analytics/wikihow-t5-small")
    return tokenizer, model

def textPreprocess(text):
    pattern = "[^\x00-\x7F]+"
    preprocess_text = text.strip().replace("\n","")
    preprocess_text = re.sub(pattern, "", preprocess_text)
    return preprocess_text

def textTokenized(text, tokenizer):
    tokenized_text = tokenizer.encode(text, return_tensor="pt")
    return tokenized_text

def modelParameters(model,text):
    summary_ids = model.generate(
        text,
        max_length = 30,
        num_beams = 2,
        repetition_penalty = 1.0,
        length_penalty = 1.0,
        early_stopping = True
    )
    return summary_ids

def output(tokenizer, summary_ids):
    return tokenizer.decode(summary_ids[0], skip_special_tokens = True)

def main():
    tokenizer, model = loadComponent()
    text = textPreprocess(text)
    text = textTokenized(text)
    param = modelParameters(model, text)
    summary = output(tokenizer, param)
    print(summary)

if __name__ == "main":
    main()


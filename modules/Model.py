#importing libraries
import numpy as np 
import pandas as pd 
import re
import torch
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModelWithLMHead

def main(text):
    tokenizer = AutoTokenizer.from_pretrained("deep-learning-analytics/wikihow-t5-small")
    model = AutoModelWithLMHead.from_pretrained("deep-learning-analytics/wikihow-t5-small")
    pattern = "[^\x00-\x7F]+"
    text = text.strip().replace("\n","")
    text = re.sub(pattern, "", text)
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    summary_ids = model.generate(
        input_ids = tokenized_text,
        max_length = 30,
        num_beams = 2,
        repetition_penalty = 2.5,
        length_penalty = 1.0,
        early_stopping = True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens = True)

# text = "Dear Reader:na beautiful late spring aftemoon, twenty-five years ago, two young men graduatedfrom the same college. They were very much alike, these two young men. Both had beenbetter than average students, both were personable and bothas young college graduatesarewere filled with ambitious dreams forthe future.Recently, these men retumed to their college for their 25th reunion.They were still very much alike. Both were happily married. Both had three children.And both, it tumed out, had gone to work for the same Midwestem manufacturingcompany after graduation, and were stil there.But there was a difference, One of the men was manager of a small department of thatcompany. The other was its president"
# summary = main(text)
# print(summary)


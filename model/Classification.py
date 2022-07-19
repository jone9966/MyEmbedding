import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Classification:
    def __init__(self, model, tokenizer, drop_out, num_class):
        self.model = model
        self.drop_out = nn.Dropout(drop_out)
    
    
        
    def forward(self, input_ids, attention_mask):
        pooler = gpt(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output = pooler['hidden_states'][-1]
        batch_size = output.shape[0]
        output = self.classifier(output.reshape(batch_size, -1))
        return output
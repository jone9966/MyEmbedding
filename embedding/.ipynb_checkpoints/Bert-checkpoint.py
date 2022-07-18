import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class BERTDataset(Dataset):
    def __init__(self, corpus, label, max_len=300):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
        self.vocab_size = self.tokenizer.vocab_size
        self.sentences = [self.transform(i) for i in corpus]
        self.labels = [np.array(i) for i in label]

    def transform(self, data):
        data = self.tokenizer(data, max_length=self.max_len, padding="max_length", truncation=True,)
        return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx] + (self.labels[idx],)


class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=8, dr_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("beomi/kcbert-base", return_dict=False)
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooler = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        out = self.dropout(pooler)
        return  self.classifier(out)


device = torch.device("mps" if torch.has_mps else 'cpu')

max_len = 512
batch_size = 4
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
learning_rate = 5e-5

df_token = pd.read_csv('/Users/zum/Dev/Dataset/train_category.csv')
corpus = [t for t in df_token['text']]
label = to_categorical(LabelEncoder().fit_transform(df_token['label']))

train_dataset = BERTDataset(corpus, label)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)


model = BERTClassifier()
model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


losses = []
for epoch in range(num_epochs):
    model.train()
    for batch_id, (input_ids, token_type_ids, attention_mask, label) in enumerate(tqdm(train_dataloader)):
        input_ids = input_ids.long().to(device)
        token_type_ids = token_type_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        label = label.to(device)

        optimizer.zero_grad()
        out = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fn(out, label)
        loss.backward()
        losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
    print(sum(losses)/len(losses))
    state = {'Epoch': epoch,
             'State_dict': model.state_dict(),
             'Optimizer': optimizer.state_dict()}


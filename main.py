from model import SentiBERT
from dataset import ReviewDataset

import sys
import os
from pathlib import Path
import pickle
import tqdm
import random
sys.path.append('..')

from transformers import AutoModel, AutoTokenizer, AutoConfig

import json
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    yelp_df = pd.DataFrame(data)

    return yelp_df


def train():
    pass


if __name__=="__main__":
    # Define parameters
    train=True
    eval = False

    model_name = 'bert-base-uncased'
    max_length = 512
    batch_size = 10
    output_path = '../data/'
    num_epoch = 4
    learning_rate = 3e-4
    batch_size = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentiBERT(model_name_or_path=model_name, hidden_dim=768).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    datafile = "../data/yelp_academic_dataset_review.json"
    data_df = load_dataset(datafile)

    dataset = ReviewDataset(data_df, tokenizer, 512, output_path)
    train_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - train_set_size
    train_dataset, test_dataset = random_split(dataset, [train_set_size, test_set_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    seed_val = 42

    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if train:
        for epoch in range(num_epoch):
            loss = 0
            total_batch = 0
            for batch, labels in train_dataloader:
                optimizer.zero_grad()

                output = model(batch.to(device))
                batch_loss = criterion(output, labels)

                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item() * batch_size
                total_batch += batch_size

            print(f'The total loss at epoch {epoch + 1} : {loss:.5f}')
            print(f'The average loss at {epoch + 1} : {loss:.5f}')

    if eval:
        model.eval()

        y_pred_list = []
        with torch.no_grad():
            testset_size = 0
            corrects = 0
            for batch, labels in test_dataloader:
                logits = model(batch.to(device))
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
                corrects += (pred == labels)
                testset_size += batch_size
            accuracy = corrects.sum().float() / float(testset_size)
            print(f'The accuracy of the testset is {accuracy:.3f}')








import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def stars_to_sentiment(x):
    return 2 if x >= 4.0 else (0 if x < 2.0 else 1)


def tokenize_text(text):
    return ' '.join([word for word in word_tokenize(text) if not word.lower() in stop_words])


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, output_path):
        self.text = df['text']
        self.labels = df['stars'].apply(lambda x: stars_to_sentiment(x))

        # Remove stop words
        filtered_text = df['text'].apply(lambda x: tokenize_text(x))

        # tokenization
        self.text = tokenizer.batch_encode_plus(filtered_text.tolist(), truncation=True,
                                                add_special_tokens=True, padding='max_length', max_length=max_length)[
            'input_ids']

        data_path = Path(output_path, 'tokenized_data.pkl')
        if not Path(data_path).exists():
            with open(data_path, 'wb') as f:
                pickle.dump({'text': self.text, 'labels': self.labels}, f)

        print(f'Loaded {len(self.labels)} examples.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.text[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


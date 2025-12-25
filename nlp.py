import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
import re

class TweetDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return seq, label
        return seq

def build_vocab_and_tokenize(df, max_len, min_df):
    """Создает словарь слов из текстовых данных, токенизирует тексты и 
       возвращает словарь, размер словаря и индекс. посл-ти"""
    counter = Counter()
    for text in df['clean_text']:
        tokens = text.split()[:max_len]
        counter.update(tokens)

    vocab = ['<pad>', '<unk>'] + [
        word for word, count in counter.items() if count >= min_df
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)

    def tokenize(text):
        return [
            word2idx.get(token, word2idx['<unk>'])
            for token in text.split()[:max_len]
        ]

    train_indices = df['clean_text'].apply(tokenize).tolist()

    return vocab, word2idx, vocab_size, train_indices

def clean_text(text):
    """очистка текста"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text)
    text = re.sub(r'@\w+', ' user ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def collate_fn(batch):
    """ формирует батч: паддит последовательности до одинаковой длины
    и собирает метки в тензор"""
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch]) if batch[0][1] is not None else None
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences_padded, labels
    
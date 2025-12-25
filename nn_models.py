import torch
import torch.nn as nn

from transformers import AutoModel
from torch.utils.data import Dataset

class BiRNNClassifier(nn.Module):
    """
    Bidirectional RNN (vanilla RNN) classifier для бинарной классификации текстов.
    
    Args:
        vocab_size (int): Размер словаря
        embed_dim (int): Размерность embedding слоя
        hidden_dim (int): Размер скрытого слоя RNN
        num_layers (int): Количество слоёв RNN
        dropout (float): Вероятность dropout
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 потому что bidirectional
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Входные индексы токенов (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Логиты классификации (batch_size, 1)
        """
        x = self.embedding(x)
        x = self.dropout(x)
        out, _ = self.rnn(x)
        # Берём последний hidden state обоих направлений
        out = out[:, -1, :]  # (batch, hidden*2)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU classifier для бинарной классификации текстов.
    
    Args:
        vocab_size (int): Размер словаря
        embed_dim (int): Размерность embedding слоя
        hidden_dim (int): Размер скрытого слоя GRU
        num_layers (int): Количество слоёв GRU
        dropout (float): Вероятность dropout
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 потому что bidirectional
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Входные индексы токенов (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Логиты классификации (batch_size, 1)
        """
        x = self.embedding(x)
        x = self.dropout(x)
        out, _ = self.gru(x)
        # Берём последний hidden state обоих направлений
        out = out[:, -1, :]  # (batch, hidden*2)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier для бинарной классификации текстов.
    
    Args:
        vocab_size (int): Размер словаря
        embed_dim (int): Размерность embedding слоя
        hidden_dim (int): Размер скрытого слоя LSTM
        num_layers (int): Количество слоёв LSTM
        dropout (float): Вероятность dropout
        pretrained_embeddings (torch.Tensor, optional): Предобученные эмбеддинги
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, (hidden, cell) = self.lstm(x)
        # последний hidden state обоих направлений
        out = out[:, -1, :]  # (batch, hidden*2)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class CNNClassifier(nn.Module):
    """
    CNN classifier для бинарной классификации текстов.
    Использует несколько сверточных фильтров с разными размерами ядра.
    
    Args:
        vocab_size (int): Размер словаря
        embed_dim (int): Размерность embedding слоя
        num_filters (int): Количество фильтров для каждого размера ядра
        filter_sizes (list): Размеры сверточных ядер (n-граммы)
        dropout (float): Вероятность dropout
        pretrained_embeddings (torch.Tensor, optional): Предобученные эмбеддинги
    """
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, 
                 filter_sizes=[3, 4, 5], dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
            embed_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
  
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len) для Conv1d
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch, num_filters, seq_len - fs + 1)
            pooled = torch.max_pool1d(conv_out, conv_out.shape[2])  # (batch, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch, num_filters)
        
        x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

class BERTweetClassifier(nn.Module):
    """
    BERTweet classifier для бинарной классификации твитов
    Используем предобученную модель vinai/bertweet-base.
    """
    def __init__(self, model_name='vinai/bertweet-base', dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

class BERTTweetDataset(Dataset):
    """
    Dataset для BERT-based моделей с токенизацией.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else -1
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

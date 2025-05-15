"""
BiRNN-Attention Model for Racial Bias Detection
==============================================
This module implements the BiRNN with Attention model architecture
for detecting racial bias in text.
"""

import torch
import torch.nn as nn

# Define device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiRNNAttention(nn.Module):
    """
    Bidirectional RNN with Attention mechanism for text classification.

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of word embeddings
        hidden_dim (int): Dimension of hidden layers in LSTM
        output_dim (int): Number of output classes
        n_layers (int): Number of LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        dropout (float): Dropout rate
        pad_idx (int): Padding token index
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, num_categories=None):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM layer
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)

        # Output layer
        self.fc_binary = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        # Output for category classification (if specified)
        self.multi_task = num_categories is not None
        if self.multi_task:
            self.fc_category = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_categories)

        self.dropout = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, text, text_lengths):
        """
        Forward pass through the network.

        Args:
            text (torch.Tensor): Input text tensor [batch_size, seq_len]
            text_lengths (torch.Tensor): Actual lengths of each sequence in the batch

        Returns:
            torch.Tensor: Model output [batch_size, output_dim]
        """
        # text = [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sent len, emb dim]

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),
                                                          batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output = [batch size, sent len, hid dim * num directions]

        # Attention mechanism
        attention_scores = self.attention(output)
        # attention_scores = [batch size, sent len, 1]

        # Create attention mask from lengths
        mask = torch.arange(output.size(1), device=device)[None, :] < text_lengths[:, None]
        # mask = [batch size, sent len]
        mask = mask.unsqueeze(-1)
        # mask = [batch size, sent len, 1]

        # Apply mask to attention scores
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)
        # attention_weights = [batch size, sent len, 1]

        # Apply attention weights to the output
        context_vector = torch.sum(attention_weights * output, dim=1)
        # context_vector = [batch size, hid dim * num directions]


        # Return both classification outputs if multi-task
        if self.multi_task:
            binary_output = self.fc_binary(self.dropout(context_vector))
            category_output = self.fc_category(self.dropout(context_vector))
            return binary_output, category_output
        else:
            return self.fc_binary(self.dropout(context_vector))


class TextDataset(torch.utils.data.Dataset):
    """
    Dataset class for text classification.

    Args:
        texts (list): List of text samples
        labels (list): List of corresponding labels
        tokenizer: HuggingFace tokenizer
        max_len (int): Maximum sequence length
    """
    def __init__(self, texts, binary_labels, tokenizer, category_labels=None, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.binary_labels = binary_labels
        self.category_labels = category_labels
        self.multi_task = category_labels is not None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        binary_labels = self.binary_labels[idx]

        encoding = self.tokenizer(text,
                                  truncation=True,
                                  max_length=self.max_len,
                                  padding='max_length',
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'binary_labels': torch.tensor(binary_labels, dtype=torch.long),
            'category_labels': torch.tensor(self.category_labels[idx], dtype=torch.long) if self.multi_task else torch.tensor(0, dtype=torch.long)
        }
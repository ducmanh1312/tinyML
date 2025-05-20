import torch
from torch import nn


# class LSTM(torch.nn.Module):
#     def __init__(self, class_num, vocab_size, embedding_dim=128, hidden_dim=768, num_layers=1, dropout=0.5):
#         super(LSTM, self).__init__()
#
#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.LSTM = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Dropout(dropout),
#             torch.nn.Linear(hidden_dim, class_num),
#             torch.nn.ReLU()
#         )
#
#     def forward(self, inputs):
#         inputs = self.word_embeddings(inputs)
#         x, _ = self.LSTM(inputs, None)
#         x = x[:, -1, :]
#         x = self.classifier(x)
#
#         return x


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=768, num_layers=1, dropout=0.5, class_num=117):
        super(LSTM, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(hidden_dim, class_num),
        #     torch.nn.ReLU()
        # )

    def forward(self, inputs):
        inputs = self.word_embeddings(inputs)
        #print(inputs.shape)
        x, _ = self.LSTM(inputs, None)
        #print(x.shape)
        # x = x[:, -1, :]
        # x = self.classifier(x)

        return x
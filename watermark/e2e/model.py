import torch
from torch import nn
from torch.cuda.amp import autocast

class Enc(nn.Module):
    def __init__(self, input_dim, mapper_layers=5, win_size=10, hidden_dim=64):
        super(Enc, self).__init__()

        # mapper
        self.mapper = nn.ModuleList()

        self.mapper.append(nn.Linear(input_dim, hidden_dim))
        self.mapper.append(nn.ReLU())

        for _ in range(mapper_layers - 1):
            self.mapper.append(nn.Linear(hidden_dim, hidden_dim))
            self.mapper.append(nn.ReLU())

        self.mapper.append(nn.Linear(hidden_dim, hidden_dim))

        # enc
        self.fc1 = nn.Linear(win_size * hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    @autocast()
    def forward(self, x):
        b, k, w, _ = x.shape

        # mapper
        for layer in self.mapper:
            x = layer(x)
        x = x.view(b, k, -1)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)

        # activate
        logit = out.squeeze(-1)
        logit_minus_mean = logit - logit.mean(dim=1, keepdim=True)
        logit_minus_mean = torch.tanh(1000 * logit_minus_mean)

        return logit_minus_mean


class Dec(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=1, num_layers=3):
        super(Dec, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_dim, num_classes)

    @autocast()
    def forward(self, x):
        b, l, _ = x.size()
        x, _ = self.lstm(x)
        x = self.fc_2(x[:, -1, :])
        x = self.relu(x)
        logit = self.fc_1(x)
        return logit
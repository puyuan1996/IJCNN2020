import torch
from torch import nn as nn
from torch.nn import functional as F


class RecurrentLatentEncoder(nn.Module):
    '''
    encode context via recurrent network
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = latent_dim*2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_size)

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.hidden_in = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).cpu(),
                          torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).cpu())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

    def forward(self, x, hidden_in):
        batch_sz, seq, feat = x.size()
        out = x.view(batch_sz, seq, feat)

        x = F.relu(self.fc1(out))
        out = F.relu(self.fc2(x))
        out = out.view(batch_sz, seq, -1)

        out, lstm_hidden = self.lstm(out, hidden_in)
        # self.hidden = lstm_hidden
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        output = self.fc3(out)
        # output = F.relu(self.fc3(out))

        return output, lstm_hidden

    # def reset(self, num_tasks=1):
    #     self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)


class RecurrentLatentEncoderSmall(nn.Module):
    '''
    encode context via recurrent network
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = latent_dim*2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        #self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_size)

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.hidden_in = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).cpu(),
                      torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).cpu())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

    def forward(self, x, hidden_in):
        batch_sz, seq, feat = x.size()
        # out = x.view(batch_sz, seq, feat)
  
        out = F.relu(self.fc1(x))
        #out = F.relu(self.fc2(x))
        out = out.view(batch_sz, seq, -1)
  
        out, lstm_hidden = self.lstm(out, hidden_in)
        # self.hidden = lstm_hidden
        # take the last hidden state to predict z
        out = out[:, -1, :]
 
        # output layer
        output = self.fc3(out)
        # output = F.relu(self.fc3(out))
  
        return output, lstm_hidden


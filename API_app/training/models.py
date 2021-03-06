import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

# LSTM models used for predicting the capacity of a cell
# There are two different models: one 
# with a covariate feature input, and one without

class Uncertain_LSTM_Module(nn.Module):
    def __init__(
        self,
        num_in=1,
        num_augment=3,
        num_hidden=100,
        seq_len=5,
        n_layers=2,
        dropout=0.0,
        num_hidden_lstm=-1,
    ):
        super(Uncertain_LSTM_Module, self).__init__()
        self.hidden_dim = num_hidden
        if num_hidden_lstm == -1:
            num_hidden_lstm = num_hidden
        self.hidden_dim_lstm = num_hidden_lstm
        self.seq_len = seq_len
        self.n_layers = n_layers

        ## LSTM layer
        self.lstm1 = nn.LSTM(num_in, self.hidden_dim_lstm, num_layers=n_layers)  #

        self.linear1 = nn.Linear(self.hidden_dim_lstm + num_augment, num_hidden)

        self.drop_layer = nn.Dropout(p=dropout)
        self.mean_state_layer = nn.Linear(num_hidden, 1)
        self.var_state_layer = nn.Linear(num_hidden, 1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.hidden_dim_lstm),
            torch.zeros(self.n_layers, self.seq_len, self.hidden_dim_lstm),
        )

    def forward(self, x, c):
        x, self.hidden = self.lstm1(x)

        x = x[:, -1].view(-1, self.hidden_dim_lstm)

        x = torch.cat((x, c), dim=1)
        self.intermediate_output = x
        x = F.relu(self.linear1(x))
        x = self.drop_layer(x)

        mean_state = torch.tanh(self.mean_state_layer(x))

        var_state = F.softplus(self.var_state_layer(x)) + 10e-6

        return (mean_state, var_state)


class Uncertain_LSTM(nn.Module):
    def __init__(
        self,
        num_in=1,
        num_augment=3,
        num_hidden=100,
        seq_len=5,
        n_layers=2,
        dropout=0.0,
        num_hidden_lstm=-1,
    ):
        super(Uncertain_LSTM, self).__init__()
        self.hidden_dim = num_hidden
        if num_hidden_lstm == -1:
            num_hidden_lstm = num_hidden
        self.hidden_dim_lstm = num_hidden_lstm
        self.seq_len = seq_len
        self.n_layers = n_layers

        ## LSTM layer
        self.lstm1 = nn.LSTM(num_in, self.hidden_dim_lstm, num_layers=n_layers)  #

        self.linear1 = nn.Linear(self.hidden_dim_lstm + num_augment, num_hidden)

        self.drop_layer = nn.Dropout(p=dropout)
        self.mean_state_layer = nn.Linear(num_hidden, 1)
        self.var_state_layer = nn.Linear(num_hidden, 1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.hidden_dim_lstm),
            torch.zeros(self.n_layers, self.seq_len, self.hidden_dim_lstm),
        )

    def forward(self, x, c):
        x, self.hidden = self.lstm1(x)

        x = x[:, -1].view(-1, self.hidden_dim_lstm)

        x = torch.cat((x, c), dim=1)
        x = F.relu(self.linear1(x))
        x = self.drop_layer(x)

        mean_state = torch.tanh(self.mean_state_layer(x))

        var_state = F.softplus(self.var_state_layer(x)) + 10e-6

        return (mean_state, var_state)


class Uncertain_LSTM_NoCovariate(Uncertain_LSTM):
    def __init__(
        self,
        num_in=1,
        num_augment=3,
        num_hidden=100,
        seq_len=5,
        n_layers=2,
        dropout=0.0,
        num_hidden_lstm=-1,
    ):
        super(Uncertain_LSTM_NoCovariate, self).__init__()
        self.hidden_dim = num_hidden
        if num_hidden_lstm == -1:
            num_hidden_lstm = num_hidden
        self.hidden_dim_lstm = num_hidden_lstm
        self.seq_len = seq_len
        self.n_layers = n_layers

        ## LSTM layer
        self.lstm1 = nn.LSTM(num_in, self.hidden_dim_lstm, num_layers=n_layers)  #

        ## Output layers
        self.linear1 = nn.Linear(self.hidden_dim_lstm, num_hidden)
        self.drop_layer = nn.Dropout(p=dropout)
        self.mean_state_layer = nn.Linear(num_hidden, 1)
        self.var_state_layer = nn.Linear(num_hidden, 1)

    def forward(self, x, c):
        x, self.hidden = self.lstm1(x)

        x = x[:, -1].view(-1, self.hidden_dim_lstm)

        x = F.relu(self.linear1(x))
        x = self.drop_layer(x)
        mean_state = torch.tanh(self.mean_state_layer(x))

        var_state = self.var_state_layer(x)

        return (mean_state, var_state)





if __name__ == "__main__":

    pass

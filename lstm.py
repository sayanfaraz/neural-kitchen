import torch.nn as nn
import torch




class LSTM(nn.Module):

    def __init__(self, input_dims, hidden_size):

        super().__init__()

        self.h = torch.randn(hidden_size)
        self.C = nn.Parameter()

        self.W_fh = torch.randn(hidden_size)
        self.W_fi = torch.randn(hidden_size)
        self.b_fh = torch.randn()
        self.b_fi = torch.randn()

        self.W_ih = torch.randn(hidden_size)
        self.W_ii = torch.randn(hidden_size)
        self.b_ih = torch.randn()
        self.b_ii = torch.randn()

        self.W_Ca_t_h = nn.Parameter()
        self.W_Ca_t_i = nn.Parameter()
        self.b_Ca_t_h = nn.Parameter()
        self.b_Ca_t_i = nn.Parameter()

        self.W_oh = nn.Parameter()
        self.W_oi = nn.Parameter()
        self.b_oh = nn.Parameter()
        self.b_oi = nn.Parameter()

    def forward(self, x):
        forget_gate = nn.Sigmoid(self.W_fh*self.h + self.b_fh +
                         self.W_fi*x + self.b_fi)

        input_gate = nn.Sigmoid(self.W_ih*self.h + self.b_ih +
                         self.W_ii*x + self.b_ii)
        C_add_t = nn.Tanh(self.W_Ca_t_h*self.h + self.b_Ca_t_h +
                          self.W_Ca_t_i*x + self.b_Ca_t_i)
        self.C = forget_gate * self.C + input_gate * C_add_t

        output_gate = nn.Sigmoid(self.W_oh*self.h + self.b_oh +
                         self.W_oi*x + self.b_oi)

        self.h = nn.Tanh(self.C) * output_gate

        return self.h


class LSTMNet(nn.Module):

    def __init__(self, input_dims):
        super().__init__()

        self.lstm_1 = LSTM(input_dims)
        self.lstm_2 = LSTM(input_dims)
        self.lstm_3 = LSTM(input_dims)

        self.decoder = nn.Linear(1, input_dims)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        h_1 = self.lstm_1.forward(x)
        h_2 = self.lstm_2.forward(h_1)

        return nn.Softmax(self.decoder(h_2))

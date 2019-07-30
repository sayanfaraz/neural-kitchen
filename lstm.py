import torch.nn as nn
import torch.tensor as tensor

x = "asdfasfaf" \
    "fas"

class LSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.h = None
        self.C = None

        self.W_fh = nn.Parameter()
        self.W_fi = nn.Parameter()
        self.b_fh = nn.Parameter()
        self.b_fi = nn.Parameter()

        self.W_ih = nn.Parameter()
        self.W_ii = nn.Parameter()
        self.b_ih = nn.Parameter()
        self.b_ii = nn.Parameter()

        self.W_Ca_t_h = nn.Parameter()
        self.W_Ca_t_i = nn.Parameter()
        self.b_Ca_t_h = nn.Parameter()
        self.b_Ca_t_i = nn.Parameter()

        self.W_oh = nn.Parameter()
        self.W_oi = nn.Parameter()
        self.b_oh = nn.Parameter()
        self.b_oi = nn.Parameter()

    def forward(self, x):
        f_t = nn.Sigmoid(self.W_fh*self.h + self.b_fh +
                         self.W_fi*x + self.b_fi)

        i_t = nn.Sigmoid(self.W_ih*self.h + self.b_ih +
                         self.W_ii*x + self.b_ii)
        C_add_t = nn.Tanh(self.W_Ca_t_h*self.h + self.b_Ca_t_h +
                          self.W_Ca_t_i*x + self.b_Ca_t_i)
        self.C = f_t * self.C + i_t * C_add_t

        o_t = nn.Sigmoid(self.W_oh*self.h + self.b_oh +
                         self.W_oi*x + self.b_oi)

        self.h = nn.Tanh(self.C) * o_t

        return self.h

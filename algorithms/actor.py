import torch
import torch.nn as nn



class Actor(nn.Module):
    def __init__(self, input_dim, hid_dim_lstm, hid_dim_v, channel_num, branch_num):
        super(Actor, self).__init__()
        self.branch_num = branch_num
        self.channel_num = channel_num
        self.lstm = nn.LSTM(input_dim, hid_dim_lstm, batch_first=True)
        #self.hidden_norm = nn.LayerNorm(hid_dim_lstm)
        self.advs = nn.ModuleList([nn.Linear(hid_dim_lstm, channel_num+1) for _ in range(branch_num)])
        self.v = nn.Sequential(nn.Linear(hid_dim_lstm, hid_dim_v), nn.ReLU(), nn.Linear(hid_dim_v,1))


    def forward(self, x, hidden=None):
        x, (h, c) = self.lstm(x, hidden)
        #h = self.hidden_norm(h)
        #c = self.hidden_norm(c)
        q = []
        v = self.v(x)
        for idx in range(self.branch_num):
            adv = self.advs[idx](x)
            q.append(adv-adv.mean()+v)
        return torch.stack(q, dim=0), (h, c)

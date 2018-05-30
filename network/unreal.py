from torch import nn
from torch.nn import functional as F
import torch
from absl import flags

FLAGS = flags.FLAGS


class LSTMConvNetwork(nn.Module):
    def __init__(self, base_conv_net, base_lstm_net, base_value_net, base_policy_net):
        super(LSTMConvNetwork, self).__init__()
        self.base_conv_net = base_conv_net
        self.base_lstm_net = base_lstm_net
        self.base_value_net = base_value_net
        self.base_policy_net = base_policy_net

    def forward(self, x, last_action_reward, lstm_state):
        x = self.base_conv_net(x)
        x, lstm_state = self.base_lstm_net(x, last_action_reward, lstm_state)
        value = self.base_value_net(x)
        pi = self.base_policy_net(x)
        return pi, value, lstm_state


class PixelChangeNetwork(nn.Module):
    def __init__(self, action_size):
        super(PixelChangeNetwork, self).__init__()
        self.fc = nn.Linear(in_features=256, out_features=9 * 9 * 32)
        self.deconv_v = nn.ConvTranspose2d(in_channels=32,
                                           out_channels=1,
                                           kernel_size=4,
                                           stride=2,
                                           padding=0
                                           )
        self.deconv_a = nn.ConvTranspose2d(in_channels=32,
                                           out_channels=action_size,
                                           kernel_size=4,
                                           stride=2,
                                           padding=0)

    def forward(self, x):
        x = self.fc(x)
        x = x.view([-1, 32, 9, 9])
        v = F.relu(self.deconv_v(x), inplace=True)
        a = F.relu(self.deconv_a(x), inplace=True)
        conv_a_mean = torch.mean(a, dim=1, keepdim=True)
        pc_q = v + a - conv_a_mean  # -1 action size 8 8
        pc_q_max = torch.max(pc_q, dim=1, keepdim=False)[0]  # -1 8 8
        return pc_q, pc_q_max


class RewardPredictionNetwork(nn.Module):
    def __init__(self):
        super(RewardPredictionNetwork, self).__init__()
        self.fc = nn.Linear(in_features=9 * 9 * 32 * 3, out_features=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, map_output):
        map_output_flat = map_output.view(1, -1)
        rp = self.softmax(self.fc(map_output_flat))
        return rp

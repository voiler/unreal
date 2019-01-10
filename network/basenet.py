from torch import nn
import torch


class BaseConvNet(nn.Module):
    def __init__(self):
        super(BaseConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=8,
                               stride=4
                               )
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=4,
                               stride=2
                               )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  # -1 64 32 32


class BaseLSTMNet(nn.Module):
    def __init__(self, action_size):
        super(BaseLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=256 + action_size + 1,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)
        self.liner = nn.Linear(2592, 256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, last_action_reward, lstm_state=None):
        x = x.view(x.size(0), -1)
        x = self.relu(self.liner(x))
        x = torch.cat([x, last_action_reward], 1)
        x = x.unsqueeze(0)
        x, lstm_state = self.lstm(x, lstm_state)
        return x.squeeze(0), lstm_state


class BaseValueNet(nn.Module):
    def __init__(self):
        super(BaseValueNet, self).__init__()
        self.value_fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.squeeze(self.value_fc(x))
        return x


class BasePolicyNet(nn.Module):
    def __init__(self, action_size):
        super(BasePolicyNet, self).__init__()
        self.policy_fc = nn.Linear(in_features=256, out_features=action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.softmax(self.policy_fc(x))
        return x

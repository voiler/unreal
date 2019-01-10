import torch
import torch.nn as nn
from network import BaseConvNet, BaseLSTMNet, BaseValueNet, BasePolicyNet
from network.unreal import PixelChangeNetwork, RewardPredictionNetwork
from common import v_wrap


class Agent(nn.Module):
    def __init__(self, thread_index, action_size, use_pixel_change, use_value_replay, use_reward_prediction,
                 pixel_change_lambda, entropy_beta, device):
        super(Agent, self).__init__()
        self._thread_index = thread_index
        self._use_pixel_change = use_pixel_change
        self._use_value_replay = use_value_replay
        self._use_reward_prediction = use_reward_prediction
        self._pixel_change_lambda = pixel_change_lambda
        self._entropy_beta = entropy_beta
        self._device = device
        self.base_conv_net = BaseConvNet()
        self.base_lstm_net = BaseLSTMNet(action_size)
        self.base_value_net = BaseValueNet()
        self.base_policy_net = BasePolicyNet(action_size)
        self._action_size = action_size
        # self.base_net = LSTMConvNetwork(base_conv_net, base_lstm_net, base_value_net, base_policy_net)
        if use_pixel_change:
            self.pc_net = PixelChangeNetwork(action_size)
        if use_reward_prediction:
            self.rp_net = RewardPredictionNetwork()
        self.to(device)
        self.reset_state()

    def _base_loss(self, base_pi, batch_a, batch_adv, batch_r, base_v):

        # [base A3C]
        # Taken action (input for network)
        log_pi = torch.log(torch.clamp(base_pi, 1e-20, 1.0))

        # Policy entropy
        entropy = -torch.sum(base_pi * log_pi, dim=1)
        # Policy loss (output)
        policy_loss = -torch.sum(torch.sum(torch.mul(log_pi, batch_a), 1) *
                                 batch_adv + entropy * self._entropy_beta)
        # R (input for value target)

        # Value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        value_loss = 0.5 * torch.sum((batch_r - base_v) ** 2 / 2.)
        base_loss = policy_loss + value_loss
        return base_loss

    def _pc_loss(self, batch_pc_a, pc_q, batch_pc_r):
        # [pixel change]

        # Extract Q for taken action
        pc_a_reshaped = batch_pc_a.view(-1, self._action_size, 1, 1)
        pc_qa_ = torch.mul(pc_q, pc_a_reshaped)  # -1 action size 16 16
        pc_qa = torch.sum(pc_qa_, dim=1, keepdim=False)
        # (-1, 16, 16)

        # TD target for Q

        pc_loss = self._pixel_change_lambda * torch.sum(((batch_pc_r - pc_qa) ** 2 / 2.))  # lambda * l2 loss
        return pc_loss

    def _vr_loss(self, batch_vr_r, vr_v):
        # Value loss (output)
        vr_loss = torch.sum((batch_vr_r - vr_v) ** 2 / 2.)  # l2 loss
        return vr_loss

    def _rp_loss(self, rp_c, batch_rp_c):
        # Reward prediction loss (output)
        rp_c = torch.clamp(rp_c, 1e-20, 1.0)
        rp_loss = -torch.sum(batch_rp_c * torch.log(rp_c))
        return rp_loss

    def loss(self, base_pi, batch_a, batch_adv, batch_r, base_v,
             batch_pc_a, pc_q, batch_pc_r,
             batch_vr_r, vr_v,
             rp_c, batch_rp_c):
        batch_a, batch_adv, batch_r, batch_pc_a, \
        batch_pc_r, batch_vr_r, batch_rp_c = v_wrap(batch_a, self._device), \
                                             v_wrap(batch_adv, self._device), \
                                             v_wrap(batch_r, self._device), \
                                             v_wrap(batch_pc_a, self._device), \
                                             v_wrap(batch_pc_r, self._device), \
                                             v_wrap(batch_vr_r, self._device), \
                                             v_wrap(batch_rp_c, self._device)
        loss = self._base_loss(base_pi, batch_a, batch_adv, batch_r, base_v)
        print("base loss: {:.6}".format(loss.item()), end=" | ")
        if self._use_pixel_change:
            pc_loss = self._pc_loss(batch_pc_a, pc_q, batch_pc_r)
            print("pc loss: {:.6}".format(pc_loss.item()), end=" | ")
            loss = loss + pc_loss

        if self._use_value_replay:
            vr_loss = self._vr_loss(batch_vr_r, vr_v)
            print("vr loss: {:.6}".format(vr_loss.item()), end=" | ")
            loss = loss + vr_loss

        if self._use_reward_prediction:
            rp_loss = self._rp_loss(rp_c, batch_rp_c)
            print("rp loss: {:.6}".format(rp_loss.item()), end=" | ")
            loss = loss + rp_loss
        print("loss: {:.6}".format(loss.item()))
        return loss

    def reset_state(self):
        self.base_lstm_state_out = None

    def forward(self, x, last_action_reward=None, base_lstm_state_out=None, name='b'):
        x = v_wrap(x, self._device)
        x = self.base_conv_net(x)
        if name == 'rp':
            rp_c = self.rp_net(x)
            return rp_c
        last_action_reward = v_wrap(last_action_reward, self._device)
        x, self.base_lstm_state_out = self.base_lstm_net(x, last_action_reward, base_lstm_state_out)
        if name == 'pc':
            pc_q, pc_q_max = self.pc_net(x)
            return pc_q, pc_q_max
        value = self.base_value_net(x)
        if name == 'vr':
            return value
        pi = self.base_policy_net(x)
        return pi, value

    def sync_from(self, net: nn.Module):
        self.load_state_dict(net.state_dict())

    def sync_to(self, net: nn.Module):
        for lp, gp in zip(self.parameters(), net.parameters()):
            gp._grad = lp.grad

# -*- coding: utf-8 -*-
import numpy as np
import time
import torch as th
from torch.nn import functional as F
from environment.environment import Environment
from .agent import Agent
from experience.experience import Experience, ExperienceFrame
from absl import flags

FLAGS = flags.FLAGS
LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class Trainer(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 env_name,
                 use_pixel_change,
                 use_value_replay,
                 use_reward_prediction,
                 pixel_change_lambda,
                 entropy_beta,
                 local_t_max,
                 gamma,
                 gamma_pc,
                 experience_history_size,
                 max_global_time_step,
                 optimizor):

        self.thread_index = thread_index
        self.env_name = env_name
        self.use_pixel_change = use_pixel_change
        self.use_value_replay = use_value_replay
        self.use_reward_prediction = use_reward_prediction
        self.local_t_max = local_t_max
        self.gamma = gamma
        self.gamma_pc = gamma_pc
        self.experience_history_size = experience_history_size
        self.max_global_time_step = max_global_time_step
        self.action_size = Environment.get_action_size(env_name)
        self.local_network = Agent(thread_index,
                                   self.action_size,
                                   use_pixel_change,
                                   use_value_replay,
                                   use_reward_prediction,
                                   pixel_change_lambda,
                                   entropy_beta)

        self.global_network = global_network
        self.experience = Experience(self.experience_history_size)
        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0
        self.optimizor = optimizor
        self.distribution = th.distributions.Categorical
        # For log output
        self.prev_local_t = 0

    def prepare(self):
        self.environment = Environment.create_environment(self.env_name)

    def close(self):
        self.environment.close()

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
                self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def _adjust_learning_rate(self, cur_learning_rate):
        for param_group in self.optimizor.param_groups:
            param_group['lr'] = cur_learning_rate

    def choose_action(self, pi):
        prob = F.softmax(pi, dim=1)
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def _record_score(self, summary_writer, score, global_t):
        summary_writer.add_scalar("score", score, global_t)

    def set_start_time(self, start_time):
        self.start_time = start_time

    def _fill_experience(self):
        """
        Fill experience buffer until buffer is full.
        """
        prev_state = self.environment.last_state
        last_action = self.environment.last_action
        last_reward = self.environment.last_reward
        last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                      self.action_size,
                                                                      last_reward)
        pi_, _ = self.local_network(
            [self.environment.last_state],
            [last_action_reward])
        action_id = pi_.detach()
        if th.cuda.is_available():
            action_id = action_id.cpu()
        action = self.choose_action(action_id)
        new_state, reward, terminal, pixel_change = self.environment.process(action)

        frame = ExperienceFrame(prev_state, reward, action, terminal, pixel_change,
                                last_action, last_reward)
        self.experience.add_frame(frame)
        if terminal:
            self.environment.reset()

        if self.experience.is_full():
            self.environment.reset()
            print("Replay buffer filled")

    def _print_log(self, global_t):
        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    def _process_base(self, global_t, summary_writer):
        # [Base A3C]
        states = []
        last_action_rewards = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        start_lstm_state = self.local_network.base_lstm_state_out

        # t_max times loop
        for _ in range(self.local_t_max):
            # Prepare last action reward
            last_action = self.environment.last_action
            last_reward = self.environment.last_reward
            last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                          self.action_size,
                                                                          last_reward)

            pi_, value_ = self.local_network(
                [self.environment.last_state],
                [last_action_reward])
            pi_, value_ = pi_.detach(), value_.detach()
            if th.cuda.is_available():
                pi_, value_estimate = pi_.cpu(), value_.cpu()
            action = self.choose_action(pi_)
            actions.append(action)
            states.append(self.environment.last_state)
            last_action_rewards.append(last_action_reward)

            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("pi={}".format(pi_.numpy()))
                print(" V={}".format(value_))

            prev_state = self.environment.last_state

            # Process game
            new_state, reward, terminal, pixel_change = self.environment.process(action)

            frame = ExperienceFrame(prev_state, reward, action, terminal, pixel_change,
                                    last_action, last_reward)

            # Store to experience
            self.experience.add_frame(frame)

            self.episode_reward += reward

            rewards.append(reward)

            self.local_t += 1

            if terminal:
                terminal_end = True
                print("score={}".format(self.episode_reward))

                self._record_score(summary_writer, self.episode_reward, global_t)

                self.episode_reward = 0
                self.environment.reset()
                self.local_network.reset_state()
                break

        R = 0.0
        if not terminal_end:
            _, R = self.local_network([new_state],
                                      [frame.get_action_reward(self.action_size)])
            R = R.detach()
            if th.cuda.is_available():
                R = R.cpu()
            R = R.numpy()
        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_adv = []
        batch_R = []

        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + self.gamma * R
            adv = R - Vi
            a = np.zeros([self.action_size], dtype=np.float32)
            a[ai] = 1.0
            batch_si.append(si)
            batch_a.append(a)
            batch_adv.append(adv)
            batch_R.append(R)

        batch_si.reverse()
        batch_a.reverse()
        batch_adv.reverse()
        batch_R.reverse()
        return batch_si, last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state

    def _process_pc(self):
        # [pixel change]
        # Sample 20+1 frame (+1 for last next state)
        pc_experience_frames = self.experience.sample_sequence(self.local_t_max + 1)
        # Reverse sequence to calculate from the last
        pc_experience_frames.reverse()

        batch_pc_si = []
        batch_pc_a = []
        batch_pc_R = []
        batch_pc_last_action_reward = []

        pc_R = np.zeros([20, 20], dtype=np.float32)
        if not pc_experience_frames[1].terminal:
            _, pc_q = self.local_network(
                [pc_experience_frames[0].state],
                [pc_experience_frames[0].get_last_action_reward(self.action_size)], name='pc')
            pc_R = pc_q.detach()
            if th.cuda.is_available():
                pc_R = pc_R.cpu()
            pc_R = pc_R.numpy()

        for frame in pc_experience_frames[1:]:
            pc_R = frame.pixel_change + self.gamma_pc * pc_R
            a = np.zeros([self.action_size], dtype=np.float32)
            a[frame.action] = 1.0
            last_action_reward = frame.get_last_action_reward(self.action_size)

            batch_pc_si.append(frame.state)
            batch_pc_a.append(a)
            batch_pc_R.append(pc_R)
            batch_pc_last_action_reward.append(last_action_reward)

        batch_pc_si.reverse()
        batch_pc_a.reverse()
        batch_pc_R.reverse()
        batch_pc_last_action_reward.reverse()

        return batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R

    def _process_vr(self):
        # [Value replay]
        # Sample 20+1 frame (+1 for last next state)
        vr_experience_frames = self.experience.sample_sequence(self.local_t_max + 1)
        # Reverse sequence to calculate from the last
        vr_experience_frames.reverse()

        batch_vr_si = []
        batch_vr_R = []
        batch_vr_last_action_reward = []

        vr_R = 0.0
        if not vr_experience_frames[1].terminal:
            _, vr_R = self.local_network(
                [vr_experience_frames[0].state],
                [vr_experience_frames[0].get_last_action_reward(self.action_size)])
            vr_R = vr_R.detach()
            if th.cuda.is_available():
                vr_R = vr_R.cpu()
            vr_R = vr_R.numpy()
        # t_max times loop
        for frame in vr_experience_frames[1:]:
            vr_R = frame.reward + self.gamma * vr_R
            batch_vr_si.append(frame.state)
            batch_vr_R.append(vr_R)
            last_action_reward = frame.get_last_action_reward(self.action_size)
            batch_vr_last_action_reward.append(last_action_reward)

        batch_vr_si.reverse()
        batch_vr_R.reverse()
        batch_vr_last_action_reward.reverse()

        return batch_vr_si, batch_vr_last_action_reward, batch_vr_R

    def _process_rp(self):
        # [Reward prediction]
        rp_experience_frames = self.experience.sample_rp_sequence()
        # 4 frames

        batch_rp_si = []
        batch_rp_c = []

        for i in range(3):
            batch_rp_si.append(rp_experience_frames[i].state)

        # one hot vector for target reward
        r = rp_experience_frames[3].reward
        rp_c = [0.0, 0.0, 0.0]
        if r == 0:
            rp_c[0] = 1.0  # zero
        elif r > 0:
            rp_c[1] = 1.0  # positive
        else:
            rp_c[2] = 1.0  # negative
        batch_rp_c.append(rp_c)
        return batch_rp_si, batch_rp_c

    def process(self, global_t, summary_writer):
        # Fill experience replay buffer
        if not self.experience.is_full():
            # print("fill ")
            self.local_network.eval()
            self._fill_experience()
            return 0

        start_local_t = self.local_t

        cur_learning_rate = self._anneal_learning_rate(global_t)

        # [Base]
        batch_si, batch_last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state = \
            self._process_base(global_t, summary_writer)

        base_pi, base_v = self.local_network(batch_si, batch_last_action_rewards)
        loss_dict = {
            'base_pi': base_pi, 'batch_a': batch_a, 'batch_adv': batch_adv,
            'batch_r': batch_R, 'base_v': base_v, }
        # [Pixel change]
        if self.use_pixel_change:
            batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R = self._process_pc()
            pc_q, pc_q_max = self.local_network(batch_pc_si, batch_pc_last_action_reward, name='pc')
            loss_dict.update({'batch_pc_a': batch_pc_a, 'pc_q': pc_q, 'batch_pc_r': batch_pc_R})
        # [Value replay]
        if self.use_value_replay:
            batch_vr_si, batch_vr_last_action_reward, batch_vr_R = self._process_vr()
            _, vr_v = self.local_network(batch_vr_si, batch_vr_last_action_reward)
            loss_dict.update({'batch_vr_r': batch_vr_R, 'vr_v': vr_v})
        # [Reward prediction]
        if self.use_reward_prediction:
            batch_rp_si, batch_rp_c = self._process_rp()
            rp_c = self.local_network(batch_rp_si, name='rp')
            loss_dict.update({'rp_c': rp_c, 'batch_rp_c': batch_rp_c})
        self.local_network.train()
        # Calculate gradients and copy them to global network.
        self._adjust_learning_rate(cur_learning_rate)
        loss = self.local_network.loss(**loss_dict)
        self.optimizor.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.local_network.parameters(), FLAGS.grad_norm_clip)
        self.local_network.sync_to(self.global_network)
        self.optimizor.step()
        self.local_network.sync_from(self.global_network)
        self._print_log(global_t)

        # Return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t

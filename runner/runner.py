import threading
import time
import os
from tensorboardX import SummaryWriter
from agent import Saver
from agent import Agent
from agent import Trainer
from optim.shared_adam import SharedAdam
from environment.environment import Environment
from common import log_uniform


class Runner(object):
    def __init__(self, args):
        super(Runner, self).__init__()
        self.env_name = args.env_name
        self.log_path = os.path.join(args.log_path, args.env_name)
        self.max_time_step = args.max_time_step
        self.initial_alpha_low = args.initial_alpha_low
        self.initial_alpha_high = args.initial_alpha_high
        self.initial_alpha_log_rate = args.initial_alpha_log_rate
        self.use_pixel_change = args.use_pixel_change
        self.use_value_replay = args.use_value_replay
        self.use_reward_prediction = args.use_reward_prediction
        self.pixel_change_lambda = args.pixel_change_lambda
        self.entropy_beta = args.entropy_beta
        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.parallel_size = args.parallel_size
        self.local_t_max = args.parallel_size
        self.gamma = args.gamma
        self.gamma_pc = args.gamma_pc
        self.experience_history_size = args.experience_history_size
        self.save_interval_step = args.save_interval_step
        self.grad_norm_clip = args.grad_norm_clip
        initial_learning_rate = log_uniform(self.initial_alpha_low,
                                            self.initial_alpha_high,
                                            self.initial_alpha_log_rate)

        self.global_t = 0
        self.stop_requested = False
        self.terminate_reqested = False
        action_size = Environment.get_action_size(self.env_name)
        self.global_network = Agent(-1,
                                    action_size,
                                    self.use_pixel_change,
                                    self.use_value_replay,
                                    self.use_reward_prediction,
                                    self.pixel_change_lambda,
                                    self.entropy_beta,
                                    args.device)
        self.trainers = []
        self.global_network.share_memory()
        optimizor = SharedAdam(params=self.global_network.parameters(), lr=initial_learning_rate,
                               weight_decay=self.alpha,
                               eps=self.epsilon)

        for i in range(self.parallel_size):
            trainer = Trainer(i,
                              self.global_network,
                              initial_learning_rate,
                              self.env_name,
                              self.use_pixel_change,
                              self.use_value_replay,
                              self.use_reward_prediction,
                              self.pixel_change_lambda,
                              self.entropy_beta,
                              self.local_t_max,
                              self.gamma,
                              self.gamma_pc,
                              self.experience_history_size,
                              self.max_time_step,
                              self.grad_norm_clip,
                              optimizor,
                              args.device)
            self.trainers.append(trainer)

        self.summary_writer = SummaryWriter(self.log_path)
        self.saver = Saver(self.log_path)

        self.global_t, wall_t = self.saver.restore(self.global_network)
        # set global step
        print(">>> global step set: ", self.global_t)
        self.next_save_steps = self.save_interval_step if self.global_t == 0 \
            else (self.global_t + self.save_interval_step) // self.save_interval_step ** 2
        self.train_threads = []
        for i in range(self.parallel_size):
            self.train_threads.append(threading.Thread(target=self.train_function, args=(i, True)))
        self.start_time = time.time() - wall_t

    def train_function(self, parallel_index, preparing):
        """ Train each environment. """

        trainer = self.trainers[parallel_index]
        if preparing:
            trainer.prepare()

        # set start_time
        trainer.set_start_time(self.start_time)

        while True:
            if self.stop_requested:
                break
            if self.terminate_reqested:
                trainer.close()
                break
            if self.global_t > self.max_time_step * 1000:
                trainer.close()
                break
            if parallel_index == 0 and self.global_t > self.next_save_steps:
                # Save checkpoint
                self.save()

            diff_global_t = trainer.process(self.global_t, self.summary_writer)
            self.global_t += diff_global_t

    def run(self):
        for t in self.train_threads:
            t.start()

    def save(self):
        """ Save checkpoint.
        Called from therad-0.
        """
        self.stop_requested = True

        # Wait for all other threads to stop
        for (i, t) in enumerate(self.train_threads):
            if i != 0:
                t.join()

        # Write wall time
        wall_t = time.time() - self.start_time

        print('Start saving.')
        self.saver.save(self.global_network, self.global_t, wall_t)
        print('End saving.')

        self.stop_requested = False
        self.next_save_steps += self.save_interval_step

        # Restart other threads
        for i in range(self.parallel_size):
            if i != 0:
                thread = threading.Thread(target=self.train_function, args=(i, False))
                self.train_threads[i] = thread
                thread.start()

import threading
import math
import time
from tensorboardX import SummaryWriter
from agent import Saver
from agent import Agent
from agent import Trainer
from optim.shared_adam import SharedAdam
from environment.environment import Environment
from .runner import BaseRunner
from absl import flags

FLAGS = flags.FLAGS


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


class Runner(BaseRunner):
    def __init__(self, env_name, log_path, model_path):
        super(Runner, self).__init__()
        self.env_name = env_name
        self.log_path = log_path
        self.model_path = model_path

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
            if self.global_t > FLAGS.max_time_step * 1000:
                trainer.close()
                break
            if parallel_index == 0 and self.global_t > self.next_save_steps:
                # Save checkpoint
                self.save()

            diff_global_t = trainer.process(self.global_t, self.summary_writer)
            self.global_t += diff_global_t

    def run(self):
        initial_learning_rate = log_uniform(FLAGS.initial_alpha_low,
                                            FLAGS.initial_alpha_high,
                                            FLAGS.initial_alpha_log_rate)

        self.global_t = 0

        self.stop_requested = False
        self.terminate_reqested = False
        action_size = Environment.get_action_size(self.env_name)
        self.global_network = Agent(-1,
                                    action_size,
                                    FLAGS.use_pixel_change,
                                    FLAGS.use_value_replay,
                                    FLAGS.use_reward_prediction,
                                    FLAGS.pixel_change_lambda,
                                    FLAGS.entropy_beta)
        self.trainers = []
        self.global_network.share_memory()
        optimizor = SharedAdam(params=self.global_network.parameters(), lr=initial_learning_rate,
                               weight_decay=FLAGS.alpha,
                               eps=FLAGS.epsilon)

        for i in range(FLAGS.parallel_size):
            trainer = Trainer(i,
                              self.global_network,
                              initial_learning_rate,
                              self.env_name,
                              FLAGS.use_pixel_change,
                              FLAGS.use_value_replay,
                              FLAGS.use_reward_prediction,
                              FLAGS.pixel_change_lambda,
                              FLAGS.entropy_beta,
                              FLAGS.local_t_max,
                              FLAGS.gamma,
                              FLAGS.gamma_pc,
                              FLAGS.experience_history_size,
                              FLAGS.max_time_step,
                              optimizor)
            self.trainers.append(trainer)

        self.summary_writer = SummaryWriter(self.log_path)
        # # init or load checkpoint with saver
        self.saver = Saver(self.global_network, self.model_path)

        # checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
        if self.saver.path:
            self.global_t = self.saver.restore()
            # set global step
            print(">>> global step set: ", self.global_t)
            # set wall time
            wall_t_fname = self.model_path + '/' + 'wall_t.' + str(self.global_t)
            with open(wall_t_fname, 'r') as f:
                self.wall_t = float(f.read())
                self.next_save_steps = (self.global_t + FLAGS.save_interval_step) // FLAGS.save_interval_step ** 2

        else:
            print("Could not find old checkpoint")
            # set wall time
            self.wall_t = 0.0
            self.next_save_steps = FLAGS.save_interval_step

        # run training threads
        self.train_threads = []
        for i in range(FLAGS.parallel_size):
            self.train_threads.append(threading.Thread(target=self.train_function, args=(i, True)))

        # set start time
        self.start_time = time.time() - self.wall_t

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
        wall_t_fname = self.model_path + '/' + 'wall_t.' + str(self.global_t)
        with open(wall_t_fname, 'w') as f:
            f.write(str(wall_t))

        print('Start saving.')
        self.saver.save(self.global_t)
        print('End saving.')

        self.stop_requested = False
        self.next_save_steps += FLAGS.save_interval_step

        # Restart other threads
        for i in range(FLAGS.parallel_size):
            if i != 0:
                thread = threading.Thread(target=self.train_function, args=(i, False))
                self.train_threads[i] = thread
                thread.start()

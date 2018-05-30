import os
import sys

from absl import flags
from runner.unreal_runner import Runner
FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "Breakout-v0", "Name of a map to use.")
flags.DEFINE_integer("parallel_size", 2, "Number of environments to run in parallel")
flags.DEFINE_string("model_path", "./models", "Path for agent checkpoints")
flags.DEFINE_string("log_path", "./log", "Path for tensorboard summaries")
flags.DEFINE_integer("max_time_step", 850,
                     "Number of training batches to run in thousands")
flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")
flags.DEFINE_float("gamma", 0.95, "Reward-discount for the agent")
flags.DEFINE_float("grad_norm_clip", 40.0, "good value might depend on the environment")
flags.DEFINE_integer("local_t_max", 20,
                     "Number of steps per batch, if None use 8 for a2c and 128 for ppo")
flags.DEFINE_float("initial_alpha_low", 1e-4, "log_uniform low limit for learning rate")
flags.DEFINE_float("initial_alpha_high", 5e-3, "log_uniform high limit for learning rate")
flags.DEFINE_float("initial_alpha_log_rate", 0.5, "log_uniform interpolate rate for learning rate")
flags.DEFINE_boolean("use_pixel_change", True, "whether to use pixel change")
flags.DEFINE_boolean("use_value_replay", True, "whether to use value function replay")
flags.DEFINE_boolean("use_reward_prediction", True, "whether to use reward prediction")
flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant")
flags.DEFINE_float("pixel_change_lambda", 0.01, "pixel change lambda")
flags.DEFINE_float("gamma_pc", 0.9, "discount factor for pixel control")
flags.DEFINE_integer("experience_history_size", 2000, "experience replay buffer size")
flags.DEFINE_integer("save_interval_step", 100 * 1000, "saving interval steps")
flags.DEFINE_float("alpha", 0.99, "decay parameter for rmsprop")
flags.DEFINE_float("epsilon", 0.1, "epsilon parameter for rmsprop")


FLAGS(sys.argv)


def main():
    model_path = os.path.join(FLAGS.model_path, FLAGS.env_name)
    full_log_path = os.path.join(FLAGS.log_path, FLAGS.env_name)
    runner = Runner(FLAGS.env_name, full_log_path, model_path)
    runner.run()


if __name__ == '__main__':
    main()

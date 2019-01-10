import torch
import random
import argparse
from runner.runner import Runner

parser = argparse.ArgumentParser(description='UNREAL')
parser.add_argument("--env_name", type=str, default="Breakout-v0", help="Name of a map to use.")
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument("--parallel_size", type=int, default=2, help="Number of environments to run in parallel")
parser.add_argument("--log_path", type=str, default="./log", help="Path for tensorboard summaries")
parser.add_argument("--max_time_step", type=int, default=850,
                    help="Number of training batches to run in thousands")
parser.add_argument("--max_gradient_norm", type=float, default=500.0, help="good value might depend on the environment")
parser.add_argument("--gamma", type=float, default=0.95, help="Reward-discount for the agent")
parser.add_argument("--grad_norm_clip", type=float, default=40.0, help="good value might depend on the environment")
parser.add_argument("--local_t_max", type=int, default=20,
                    help="Number of steps per batch, if None use 8 for a2c and 128 for ppo")
parser.add_argument("--initial_alpha_low", type=float, default=1e-4, help="log_uniform low limit for learning rate")
parser.add_argument("--initial_alpha_high", type=float, default=5e-3, help="log_uniform high limit for learning rate")
parser.add_argument("--initial_alpha_log_rate", type=float, default=0.5,
                    help="log_uniform interpolate rate for learning rate")
parser.add_argument("--use_pixel_change", default=True, help="whether to use pixel change")
parser.add_argument("--use_value_replay", default=True, help="whether to use value function replay")
parser.add_argument("--use_reward_prediction", default=True, help="whether to use reward prediction")
parser.add_argument("--entropy_beta", type=float, default=0.001, help="entropy regularization constant")
parser.add_argument("--pixel_change_lambda", type=float, default=0.01, help="pixel change lambda")
parser.add_argument("--gamma_pc", type=float, default=0.9, help="discount factor for pixel control")
parser.add_argument("--experience_history_size", type=int, default=2000, help="experience replay buffer size")
parser.add_argument("--save_interval_step", type=int, default=100 * 1000, help="saving interval steps")
parser.add_argument("--alpha", type=float, default=0.99, help="decay parameter for rmsprop")
parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon parameter for rmsprop")
parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')

if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
    else:
        args.device = torch.device('cpu')
    runner = Runner(args)
    runner.run()

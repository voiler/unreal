import numpy as np
import torch as th


class Mode:
    DQN = 'dqn'
    A2C = "a2c"
    PPO = "ppo"
    UNREAL = "agent"


def weighted_random_sample(weights):
    """
    :param weights: 2d tensor [n, d] containing positive weights for sampling
    :return: 1d tensor [n] with idx in [0, d) randomly sampled proportional to weights
    """

    u = th.Tensor(weights.size()).uniform_(0, 1)
    if th.cuda.is_available():
        u = u.cuda()
    x = th.argmax(th.log(u) / weights, 1)
    return x


def select_from_each_row(params, indices):
    """
    :param params: 2d tensor of shape [d1,d2]
    :param indices: 1d tensor of shape [d1] with values in [d1, d2)
    :return: 1d tensor of shape [d1] which has one value from each row of params selected with indices
    """
    indices = indices.long()
    if th.cuda.is_available():
        indices = indices.cuda()
    indices = indices.resize_(indices.shape[0], 1)
    return th.gather(params, 1, indices)


def calculate_n_step_reward(
        one_step_rewards: np.ndarray,
        discount: float,
        last_state_values: np.ndarray):
    """
    :param one_step_rewards: [n_env, n_timesteps]
    :param discount: scalar discount paramater
    :param last_state_values: [n_env], bootstrap from these if not done
    :return:
    """

    discount = discount ** np.arange(one_step_rewards.shape[1], -1, -1)
    reverse_rewards = np.c_[one_step_rewards, last_state_values][:, ::-1]
    full_discounted_reverse_rewards = reverse_rewards * discount
    return (np.cumsum(full_discounted_reverse_rewards, axis=1) / discount)[:, :0:-1]


def general_n_step_advantage(
        one_step_rewards: np.ndarray,
        value_estimates: np.ndarray,
        gamma: float,
        lambda_par: float
):
    """
    :param one_step_rewards: [n_env, n_timesteps]
    :param value_estimates: [n_env, n_timesteps + 1]
    :param discount: "gamma" in https://arxiv.org/pdf/1707.06347.pdf and most of the rl-literature
    :param lambda_par: lambda in https://arxiv.org/pdf/1707.06347.pdf
    :return:
    """
    assert 0.0 < gamma <= 1.0
    assert 0.0 <= lambda_par <= 1.0
    batch_size, timesteps = one_step_rewards.shape
    assert value_estimates.shape == (batch_size, timesteps + 1)
    delta = one_step_rewards + gamma * value_estimates[:, 1:] - value_estimates[:, :-1]

    if lambda_par == 0:
        return delta

    delta_rev = delta[:, ::-1]
    adjustment = (gamma * lambda_par) ** np.arange(timesteps, 0, -1)
    advantage = (np.cumsum(delta_rev * adjustment, axis=1) / adjustment)[:, ::-1]
    return advantage


def combine_first_dimensions(x: np.ndarray):
    """
    :param x: array of [batch_size, time, ...]
    :returns array of [batch_size * time, ...]
    """
    first_dim = x.shape[0] * x.shape[1]
    other_dims = x.shape[2:]
    dims = (first_dim,) + other_dims
    return x.reshape(*dims)


def ravel_index_pairs(idx_pairs, n_col):
    '''
    :param idx_pairs:  2D tensor [pairs_num, 2]
    :param n_col: int
    :return:
    '''
    if th.cuda.is_available():
        return th.sum(idx_pairs * th.from_numpy(np.array([n_col, 1])[np.newaxis, ...]).float().cuda(), 1)
    else:
        return th.sum(idx_pairs * th.from_numpy(np.array([n_col, 1])[np.newaxis, ...]).float(), 1)


def dict_of_lists_to_list_of_dicts(x: dict):
    dim = {len(v) for v in x.values()}
    assert len(dim) == 1
    dim = dim.pop()
    return [{k: x[k][i] for k in x} for i in range(dim)]


def one_hot_encoding(x, num, shape):
    batch_size = x.size(0)
    x = th.unsqueeze(x, 1)
    res = th.zeros((batch_size,
                    num,
                    shape[0], shape[1]))
    if th.cuda.is_available():
        res = res.cuda()
    return res.scatter_(1, x, 1)[:, 1:, :, :]


def subsample(a, average_width):
    s = a.shape
    sh = s[0] // average_width, average_width, s[1] // average_width, average_width
    return a.reshape(sh).mean(-1).mean(1)


def calc_pixel_change(state, last_state):
    d = np.absolute(state[2:-2, 2:-2, :] - last_state[2:-2, 2:-2, :])
    # (80,80,3)
    m = np.mean(d, 2)
    c = subsample(m, 4)
    return c

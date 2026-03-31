import numpy as np
import torch
import copy

class ReplayBuffer:
    def __init__(self, args, device):
        self.device = device
        self.N = args.n_agents
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.max_steps
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.current_size = 0

        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim]),
                       'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'ca_n': np.zeros([self.buffer_size, self.episode_limit, self.N, self.N]),
                       'ap_n': np.zeros([self.buffer_size, self.episode_limit, self.N, self.action_dim]),
                       'r': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'n_s': np.zeros([self.buffer_size, self.episode_limit, self.N, self.obs_dim]),
                       'd': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'active': np.zeros([self.buffer_size, self.episode_limit, 1])
                       }  #
        self.episode_len = np.zeros(self.buffer_size)

    def store_transition(self, episode_step, obs_n, a_n, ca_n, ap_n, r, n_s, d):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n

        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['ca_n'][self.episode_num][episode_step] = ca_n
        self.buffer['ap_n'][self.episode_num][episode_step] = ap_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['d'][self.episode_num][episode_step] = d
        self.buffer['n_s'][self.episode_num][episode_step] = n_s

        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_step(self, episode_step, obs_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n


        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}

        for key in self.buffer.keys():
            #if key == 'obs_n' or key == 's' or key == 'last_onehot_a_n':
            if key == 'obs_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32, device='cpu')
            elif key == 'a_n' or key == 'ca_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.long, device='cpu')
            elif key == 'ap_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32, device='cpu')
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32, device='cpu')

        return batch, max_episode_len

    def clear_buffer(self):
        for key in self.buffer.keys():
            self.buffer[key].fill(0)

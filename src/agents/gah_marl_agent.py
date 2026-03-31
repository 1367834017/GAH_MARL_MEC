import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
import numpy as np

from src.utils.replay_buffer import ReplayBuffer




import torch


def apply_es_aware_top_k_mask(comm_logits, es_assignments, w, hat_w, n_agents):
    """
    Placeholder for a proprietary module that is temporarily omitted
    due to confidentiality agreements with industry partners.

    The full implementation will be released upon acceptance of the paper.
    """
    raise NotImplementedError(
        "This function is temporarily unavailable in the public version "
        "due to confidentiality agreements with industry partners."
    )


class CommModule(nn.Module):
    def __init__(self, args, input_dim):

        super(CommModule, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes

        self.encoder = nn.Linear(input_dim, self.hid_size)
        # LSTM
        self.lstm = nn.LSTMCell(self.hid_size, self.hid_size)

        if args.share_weights:
            self.comm_layer = nn.Linear(self.hid_size, self.hid_size)
            self.comm_layers = nn.ModuleList([self.comm_layer for _ in range(self.comm_passes)])
        else:
            self.comm_layers = nn.ModuleList([
                nn.Linear(self.hid_size, self.hid_size) for _ in range(self.comm_passes)
            ])


        self.comm_mask = torch.ones(self.n_agents, self.n_agents) - torch.eye(self.n_agents)

    def forward(self, x_input, info={}, hidden_state=None, cell_state=None):

        batch_size = x_input.shape[0]

        x = self.encoder(x_input)  # [batch, n_agents, hid_size]

        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(batch_size * self.n_agents, self.hid_size, device=x.device)
            cell_state = torch.zeros(batch_size * self.n_agents, self.hid_size, device=x.device)

        h = x#[bs,n_agents,hid_dim]
        for i in range(self.comm_passes):

            comm = h.unsqueeze(1).expand(-1, self.n_agents, -1, -1)#[bs,n_agents,n_agents,hid_dim]

            mask = self.comm_mask.view(1, self.n_agents, self.n_agents, 1).to(x.device)
            comm = comm * mask

            if 'comm_action' in info:
                comm_action = info['comm_action']
                if not isinstance(comm_action, torch.Tensor):
                    comm_action = torch.tensor(comm_action, dtype=torch.float32, device=x.device)
                else:
                    comm_action = comm_action.to(dtype=torch.float32, device=x.device)
            else:
                comm_action = torch.ones(batch_size, self.n_agents, self.n_agents, device=x.device)


            comm_action_mask = comm_action.unsqueeze(-1)  # [batch, n_agents, n_agents, 1]
            with torch.no_grad():
                comm = comm * comm_action_mask# [batch, n_agents, n_agents, hid_size]

            denom = comm_action.sum(dim=2, keepdim=True) + 1e-6
            comm_sum = comm.sum(dim=2) / denom  # [bs, n_agents, hid_size]

            c = self.comm_layers[i](comm_sum)

            h = h + c  # [batch, n_agents, hid_size]

            h_reshape = h.reshape(batch_size * self.n_agents, self.hid_size)
            hidden_state, cell_state = self.lstm(h_reshape, (hidden_state, cell_state))
            h = hidden_state.view(batch_size, self.n_agents, self.hid_size)

        return h, (hidden_state, cell_state)

class QActorWithComm(nn.Module):
    def __init__(self, n_actions, comm_module):
        super(QActorWithComm, self).__init__()
        self.comm_module = comm_module
        self.output_layer = nn.Linear(comm_module.hid_size, n_actions)

        self.comm_action_head = nn.Sequential(
            nn.Linear(2 * comm_module.hid_size, comm_module.hid_size),
            nn.ReLU(),
            nn.Linear(comm_module.hid_size, 2)
        )


    def forward(self, obs, info={}, hidden_state=None, cell_state=None):

        h, (hidden_state, cell_state) = self.comm_module(
            obs, info, hidden_state, cell_state
        )

        # 输出 Q 值
        q_values = self.output_layer(h)

        sender_h = h.unsqueeze(2).expand(-1, -1, self.comm_module.n_agents, -1)  # [bs, sender, receiver, hid]
        receiver_h = h.unsqueeze(1).expand(-1, self.comm_module.n_agents, -1, -1)  # [bs, sender, receiver, hid]

        pairwise = torch.cat([sender_h, receiver_h], dim=-1)  # [bs, n_agents, n_agents, 2*hid]

        comm_logits = self.comm_action_head(pairwise)  # [bs, n_agents, n_agents, 2]

        comm_logits = (comm_logits + comm_logits.transpose(1, 2)) / 2

        return q_values, comm_logits, (hidden_state, cell_state)



class ActorParamWithComm(nn.Module):
    def __init__(self, discrete_action_dim, param_dim, comm_module):
        super(ActorParamWithComm, self).__init__()
        self.comm_module = comm_module
        self.param_dim = param_dim
        self.n_actions = discrete_action_dim
        self.output_layer = nn.Linear(comm_module.hid_size + discrete_action_dim, param_dim)

    def forward(self, obs, k, info={}, hidden_state=None, cell_state=None):

        k_onehot = F.one_hot(k, num_classes=self.n_actions).float()  # [batch, n_agents, n_actions]

        h, (hidden_state, cell_state) = self.comm_module(
            obs, info, hidden_state, cell_state
        )
        obs_input = torch.cat([h, k_onehot], dim=-1)  # [batch, n_agents, obs_dim + n_actions]

        param = torch.sigmoid(self.output_layer(obs_input))

        return param, (hidden_state, cell_state)



class SharedMultiHeadCritic(nn.Module):
    def __init__(self, n_agents, obs_dim, discrete_action_dim, param_dim, hidden_dim=128):
        super(SharedMultiHeadCritic, self).__init__()
        self.n_agents = n_agents
        self.input_dim = obs_dim + discrete_action_dim + param_dim + n_agents * n_agents
        self.discrete_action_dim = discrete_action_dim

        self.shared_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU()
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_agents)
        ])

    def forward(self, obs_all, k_all, x_all, comm_action):

        bs = obs_all.size(0)

        k_onehot = F.one_hot(k_all, num_classes=self.discrete_action_dim).float()

        comm_input = comm_action.reshape(comm_action.shape[0], -1)  # [bs, n_agents*n_agents]

        comm_input = comm_input.unsqueeze(1).expand(-1, self.n_agents, -1)

        inp = torch.cat([obs_all, k_onehot, x_all, comm_input], dim=-1)  # [bs, n_agents, input_dim]

        inp_flat = inp.view(bs * self.n_agents, -1)
        feat_flat = self.shared_net(inp_flat)  # (bs * n_agents, hidden)

        q_list = []
        for i, head in enumerate(self.heads):
            # 取出该 agent 的 feature (bs, hidden)
            feat_i = feat_flat.view(bs, self.n_agents, -1)[:, i, :]  # (bs, hidden)
            q_i = head(feat_i).unsqueeze(1)  # (bs, 1, 1)
            q_list.append(q_i)

        q_values = torch.cat(q_list, dim=1)  # (bs, n_agents, 1)
        return q_values



# === Mixing Network for QMIX (high-level) ===
class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()
        self.N = args.n_agents
        self.state_dim = args.state_dim
        self.batch_size = args.batch_size
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.hyper_layers_num = args.hyper_layers_num

        if self.hyper_layers_num == 2:
            self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1))
        elif self.hyper_layers_num == 1:
            self.hyper_w1 = nn.Linear(self.state_dim, self.N * self.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.state_dim, self.qmix_hidden_dim * 1)
        else:
            print("wrong!!!")

        self.hyper_b1 = nn.Linear(self.state_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))


        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, q, s):
        # q.shape(batch_size, max_episode_len, N)
        # s.shape(batch_size, max_episode_len,state_dim)
        q = q.view(-1, 1, self.N)  # (batch_size * max_episode_len, 1, N)
        s = s.reshape(-1, self.state_dim)  # (batch_size * max_episode_len, state_dim)

        w1 = torch.abs(self.hyper_w1(s))  # (batch_size * max_episode_len, N * qmix_hidden_dim)
        b1 = self.hyper_b1(s)  # (batch_size * max_episode_len, qmix_hidden_dim)
        w1 = w1.view(-1, self.N, self.qmix_hidden_dim)  # (batch_size * max_episode_len, N,  qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        # torch.bmm: 3 dimensional tensor multiplication
        q_hidden = F.elu(torch.bmm(q, w1) + b1)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(s))  # (batch_size * max_episode_len, qmix_hidden_dim * 1)
        b2 = self.hyper_b2(s)  # (batch_size * max_episode_len,1)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # (batch_size * max_episode_len, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (batch_size * max_episode_len, 1， 1)

        q_total = torch.bmm(q_hidden, w2) + b2  # (batch_size * max_episode_len, 1， 1)
        q_total = q_total.view(self.batch_size, -1, 1)  # (batch_size, max_episode_len, 1)
        return q_total

class OrnsteinUhlenbeckActionNoise(object):
    """
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    Source: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, random_machine=np.random):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.random = random_machine
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(*self.X.shape)#生成与self.X相同shape的正态分布噪声
        self.X = self.X + dx
        return self.X


class GAHAgentWithComm:
    def __init__(self, obs_dim, state_dim, action_dim, action_space, obs_normal, s_normal, args, device):
        self.args = args
        self.n_agents = args.n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = action_dim
        self.action_space = action_space
        self.obs_normalization = obs_normal
        self.s_normalization = s_normal
        self.device = device
        self.comm_module = CommModule(args, obs_dim)
        self.clip_grad = args.clip_grad
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, self.n_actions + 1)])

        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.replay_memory = ReplayBuffer(self.args, self.device)

        self.noise = OrnsteinUhlenbeckActionNoise(
            (self.n_agents, self.action_parameter_size),
            mu=0,
            theta=0.15,
            sigma=0.05
        )

        self.q_actor = QActorWithComm(self.n_actions, self.comm_module).to(device)
        self.actor_param = ActorParamWithComm(self.n_actions, self.action_parameter_size, self.comm_module).to(device)
        self.critic = SharedMultiHeadCritic(self.n_agents, obs_dim, self.n_actions, self.action_parameter_size).to(device)
        self.q_actor_target = QActorWithComm(self.n_actions, self.comm_module).to(device)
        self.q_actor_target.load_state_dict(self.q_actor.state_dict())

        self.actor_param_target = ActorParamWithComm(self.n_actions, self.action_parameter_size, self.comm_module).to(device)
        self.actor_param_target.load_state_dict(self.actor_param.state_dict())

        self.critic_target = SharedMultiHeadCritic(self.n_agents, obs_dim, self.n_actions, self.action_parameter_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_params = list(self.critic.parameters())

        self.mixer = QMixer(args).to(device)
        self.target_mixer = QMixer(args).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.critic_params += list(self.mixer.parameters())

        self.q_optim = torch.optim.Adam(self.q_actor.parameters(), lr=args.lr_q)
        self.param_optim = torch.optim.Adam(self.actor_param.parameters(), lr=args.lr_param)

        self.critic_optim = torch.optim.Adam(self.critic_params, lr=args.lr_critic)
        self.loss_fn = nn.MSELoss()

        self.prev_hidden = None
        self.prev_cell = None


    def act(self, obs, epsilon, info=None):
        obs = obs.unsqueeze(0).to(self.device)
        batch_size = obs.size(0)
        if getattr(self, 'prev_hidden', None) is None:
            self.prev_hidden = torch.zeros(batch_size * self.n_agents, self.args.hid_size, device=self.device)
            self.prev_cell = torch.zeros(batch_size * self.n_agents, self.args.hid_size, device=self.device)
            self.prev_action = torch.zeros(batch_size, self.n_agents, dtype=torch.long, device=self.device)

        with torch.no_grad():
            q_values, comm_logits, (self.prev_hidden, self.prev_cell) = self.q_actor(obs, info={}, hidden_state=self.prev_hidden, cell_state=self.prev_cell)
            if np.random.rand() < epsilon:
                action = np.random.randint(q_values.size(-1), size=(self.n_agents,))

            else:
                action = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()
            action_tensor = torch.as_tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)

            if hasattr(self.args, 'w') and hasattr(self.args, 'hat_w'):

                comm_logits = apply_es_aware_top_k_mask(
                    comm_logits=comm_logits,
                    es_assignments=self.prev_action,
                    w=self.args.w,
                    hat_w=self.args.hat_w,
                    n_agents=self.n_agents
                )

            self.prev_action = action_tensor.clone()

            comm_action_gumbel = F.gumbel_softmax(comm_logits.view(-1, 2), tau=self.args.temp,
                                                  hard=True)  # [n_agents*n_agents, 2]
            comm_action = comm_action_gumbel.argmax(dim=-1).view(self.n_agents, self.n_agents).unsqueeze(0)

            comm_action[:, torch.arange(self.n_agents), torch.arange(self.n_agents)] = 1


            param, _ = self.actor_param(obs, action_tensor, info={'comm_action': comm_action},
            hidden_state=self.prev_hidden, cell_state=self.prev_cell)
            param = param.squeeze(0)
            param += torch.randn_like(param) * 1e-2

        return action, comm_action.cpu().numpy(), param.cpu().numpy()

    def update_asynchronous(self, step_count, high_step, low_step, high_freq, low_freq=1):
        """
            Placeholder for a proprietary module that is temporarily omitted
            due to confidentiality agreements with industry partners.

            The full implementation will be released upon acceptance of the paper.
            """
        raise NotImplementedError(
            "This function is temporarily unavailable in the public version "
            "due to confidentiality agreements with industry partners."
        )

    def _soft_update(self, target, source, tau):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


    def save_models(self, path):
        torch.save({
            'q_actor': self.q_actor.state_dict(),
            'actor_param': self.actor_param.state_dict()
        }, path)
        print(f"Models saved successfully to {path}")


    def load_models(self, path, map_location):
        checkpoint = torch.load(path, map_location=map_location)
        self.q_actor.load_state_dict(checkpoint['q_actor'])
        self.actor_param.load_state_dict(checkpoint['actor_param'])
        print(f"Models loaded successfully from {path}")

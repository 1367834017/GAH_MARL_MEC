import torch
import numpy as np
import argparse


from src.agents.gah_marl_agent import GAHAgentWithComm
from src.utils.normalization import Normalization
from envs.mec_env import Env

parser = argparse.ArgumentParser("Hyperparameter Setting for GAH_MARL in MEC environment")

parser.add_argument("--episode", default=2000, help='Number of epsiodes.', type=int)
parser.add_argument("--max_steps", default=500, help='Number of steps.', type=int)
parser.add_argument("--epsilon", type=float, default=1, help="Initial epsilon")
parser.add_argument("--lower_warmup_steps", type=int, default=150000, help="Number of steps training the low-level networks")
parser.add_argument("--high_freq", type=int, default=8, help="Frequency of upper-level training")
parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
parser.add_argument("--epsilon_decay", type=float, default=0.99, help="How many steps before the epsilon decays to the minimum")
parser.add_argument('--buffer-size', default=1000, help='Replay memory size in transitions.', type=int)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
parser.add_argument("--lr_q", type=float, default=1e-4,help="Learning rate of Actor")
parser.add_argument("--lr_critic", type=float, default=1e-5, help="Learning rate of Critic")
parser.add_argument("--lr_param", type=float, default=2e-4, help="Learning rate of Actor Param")
parser.add_argument("--gamma", type=float, default=0.96, help="Discount factor")
parser.add_argument("--hid_size", type=int, default=128, help="The dimension of the hidden layer of the Actor/Param/Critic network")
parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
parser.add_argument("--hyper_layers_num", type=int, default=2, help="The number of layers of hyper-network")
parser.add_argument('--layers', default='[128,]', help='Duplicate action-parameter inputs.')
parser.add_argument('--clip-grad', default=1.0, help="Parameter gradient clipping limit.", type=float)
parser.add_argument("--use_lr_decay", type=bool, default=True, help="use lr decay")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
parser.add_argument("--use_obs_norm", type=bool, default=True, help="Whether to use observation normalization")
parser.add_argument('--tau', default=0.05, help='Soft target network update averaging factor.', type=float)
parser.add_argument('--temp', default=0.5, help='Soft target network update averaging factor.', type=float)
parser.add_argument('--lambda_entropy', default=0.01, help='Soft target network update averaging factor.', type=float)

parser.add_argument('--hat_w', default=1, help='Soft target network update averaging factor.', type=float)
parser.add_argument('--w', default=2, help='Soft target network update averaging factor.', type=float)

parser.add_argument('--comm_passes', default=1, help='Number of communiction.', type=int)
parser.add_argument("--hard_attn", type=bool, default=True, help="Whether to use hard gate")
parser.add_argument("--share_weights", type=bool, default=True, help="Whether to share weights in communication layers")
parser.add_argument("--sdn_interval", type=int, default=3, help="SDN Update interval")
parser.add_argument("--comm_type", type=str, default="proposed", help="gcn,gat,no_lstm,proposed")
args = parser.parse_args()

def run_training_asy(args, seed):
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create env
    num = 5
    server = 3
    fi_m = np.random.uniform(3, 7, server)
    fi_l = np.random.uniform(0.8, 1.5, num)
    env = Env(alpha=0.6, beta=0.4, B=10, N0=pow(10, -174 / 10) * 0.001,
                   pi=500, K=num, ser=server, fi_m=fi_m, fi_l=fi_l)

    # Create N agents
    args.n_agents = env.K
    args.obs_dim = env.single_observation_space.shape[0]
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.spaces[0].n

    obs_normalizer = None
    s_normalizer = None
    if args.use_obs_norm:
        obs_normalizer = Normalization(shape=args.obs_dim)
        s_normalizer = Normalization(shape=args.state_dim)
    agent = GAHAgentWithComm(obs_dim=args.obs_dim, state_dim=args.state_dim,
                                action_dim=args.action_dim, action_space=env.action_space, args=args,
                                obs_normal=obs_normalizer, s_normal=s_normalizer, device=device)
    returns = []
    step_count = 0
    high_step = 0
    low_step = 0
    epsilon = args.epsilon
    sdn_interval = args.sdn_interval
    for episode in range(args.episode):
        torch.cuda.empty_cache()
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        cached_obs = obs.clone()
        episode_reward = []
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        act, comm_act, act_param = agent.act(cached_obs, epsilon)

        for t in range(args.max_steps):
            next_obs, reward_i, done, _ = env.step(act, act_param)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

            if t == args.max_steps - 1:
                done[:] = [True] * len(done)
            reward = np.sum(reward_i)
            episode_reward.append(reward)
            if args.use_obs_norm:
                obs_normalizer(obs.cpu().numpy(), update=True)
                s_normalizer(obs.reshape(-1, 1).squeeze().cpu().numpy(), update=True)
            if (t + 1) % sdn_interval == 0:
                cached_next_obs = next_obs.clone()
            else:
                cached_next_obs = cached_obs.clone()

            next_act, next_comm_act, next_act_param = agent.act(cached_next_obs, epsilon)
            agent.replay_memory.store_transition(t, obs.cpu().numpy(),
                                                 act, comm_act, act_param, reward,
                                                 next_obs.squeeze().cpu().numpy(),
                                                 done)
            obs = next_obs  # 真实环境状态照常滚动
            cached_obs = cached_next_obs  # SDN 视角状态滚动
            act, comm_act, act_param = next_act, next_comm_act, next_act_param

            torch.cuda.empty_cache()
            step_count += 1
            low_step += 1
            if step_count > args.lower_warmup_steps:
                high_step += 1
            torch.cuda.empty_cache()

        returns.append(np.mean(episode_reward))
        agent.replay_memory.store_last_step(args.max_steps, obs.cpu().numpy())
        if agent.replay_memory.current_size >= args.batch_size:
            torch.autograd.set_detect_anomaly(True)
            c_loss, a_loss, q_loss = agent.update_asynchronous(step_count=step_count, high_step=high_step,
                                                               low_step=low_step, high_freq=args.high_freq)  # Training
        if episode % 10 == 0:
            print("Episode: \t{} R:\t{:0.4f}".format(episode, float(np.mean(episode_reward))))
            # print(act, act_param)
            if agent.replay_memory.current_size >= args.batch_size:
                print("Actor Loss\t{:0.4f} Critic Loss\t{:0.4f} qActor Loss\t{:0.4f}".format(a_loss, c_loss, q_loss))

        if (episode + 1) % 50 == 0:
            agent.save_models(
                path=f'Multi_Seed/GAH_{seed}.pth'
            )

    return returns

if __name__ == "__main__":
    academic_seeds = [42, 1024, 2024, 3407, 9999]
    all_seeds_returns = {}
    for current_seed in academic_seeds:
        print(f'========== Initiating Deep Training with Seed: {current_seed} ==========')
        current_returns = run_training_asy(args, current_seed)
        all_seeds_returns[f'Seed_{current_seed}'] = current_returns
        print(f'========== Training for Seed {current_seed} Successfully Concluded ==========\n')





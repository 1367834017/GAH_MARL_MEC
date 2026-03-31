import os
import torch
import numpy as np
import argparse
import pandas as pd

from agents.gah_marl_agent import GAHAgentWithComm
from normalization import Normalization
from normalization import RewardNormalizer
from envs.mec_env import Env



parser = argparse.ArgumentParser("Testing for GAH-MARL in MEC environment")

parser.add_argument("--test_episode", type=int, default=10, help="Number of test episodes")
parser.add_argument("--max_steps", type=int, default=500, help="Number of steps per episode")
parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_model.pt", help="Path to trained model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

parser.add_argument("--algorithm", type=str, default="GAH-MARL", help="Name of algorithm")
parser.add_argument("--env_name", type=str, default="MEC", help="Name of environment")

parser.add_argument("--gamma", type=float, default=0.96)
parser.add_argument("--hid_size", type=int, default=128)
parser.add_argument("--qmix_hidden_dim", type=int, default=32)
parser.add_argument("--hyper_hidden_dim", type=int, default=64)
parser.add_argument("--hyper_layers_num", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--buffer_size", type=int, default=1000)
parser.add_argument("--lr_q", type=float, default=1e-4)
parser.add_argument("--lr_critic", type=float, default=1e-5)
parser.add_argument("--lr_param", type=float, default=2e-4)
parser.add_argument("--clip_grad", type=float, default=1.0)
parser.add_argument("--tau", type=float, default=0.05)
parser.add_argument("--temp", type=float, default=0.5)
parser.add_argument("--lambda_entropy", type=float, default=0.01)
parser.add_argument("--top_k", type=float, default=3)
parser.add_argument("--hat_w", type=float, default=1)
parser.add_argument("--w", type=float, default=2)

parser.add_argument("--comm_passes", type=int, default=1)
parser.add_argument("--hard_attn", type=bool, default=True)
parser.add_argument("--share_weights", type=bool, default=True)
parser.add_argument("--comm_type", type=str, default="proposed")
parser.add_argument("--use_obs_norm", type=bool, default=True)

args = parser.parse_args()


def run_test(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num = 5
    server = 3
    fi_m = np.random.uniform(3, 7, server)
    fi_l = np.random.uniform(0.8, 1.5, num)

    env = Env(
        alpha=0.6,
        beta=0.4,
        B=10,
        N0=pow(10, -174 / 10) * 0.001,
        pi=500,
        K=num,
        ser=server,
        fi_m=fi_m,
        fi_l=fi_l
    )

    args.n_agents = env.K
    args.obs_dim = env.single_observation_space.shape[0]
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.spaces[0].n

    obs_normalizer = None
    s_normalizer = None
    if args.use_obs_norm:
        obs_normalizer = Normalization(shape=args.obs_dim)
        s_normalizer = Normalization(shape=args.state_dim)

    agent = GAHAgentWithComm(
        obs_dim=args.obs_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        action_space=env.action_space,
        obs_normal=obs_normalizer,
        s_normal=s_normalizer,
        args=args,
        device=device
    )

    agent.load_models(args.checkpoint, map_location=device)

    episode_rewards = []
    episode_latencies = []
    episode_energies = []

    for episode in range(args.test_episode):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        total_reward = 0.0
        total_latency = 0.0
        total_energy = 0.0

        agent.prev_hidden = None
        agent.prev_cell = None
        agent.prev_action = None

        for t in range(args.max_steps):
            act, comm_act, act_param = agent.act(obs, epsilon=0.0)

            next_obs, reward_i, done, info = env.step(act, act_param)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

            total_reward += float(np.sum(reward_i))
            total_latency += float(info["avg_latency"])
            total_energy += float(info["avg_energy"])

            obs = next_obs

            if all(done):
                break

        episode_rewards.append(total_reward)
        episode_latencies.append(total_latency)
        episode_energies.append(total_energy)

        print(
            f"Test Episode {episode + 1}: "
            f"Reward = {total_reward:.4f}, "
            f"Latency = {total_latency:.4f}, "
            f"Energy = {total_energy:.4f}"
        )

    print("\n========== Test Summary ==========")
    print(f"Average Reward:  {np.mean(episode_rewards):.4f}")
    print(f"Average Latency: {np.mean(episode_latencies):.4f}")
    print(f"Average Energy:  {np.mean(episode_energies):.4f}")

    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame({
        "episode_reward": episode_rewards,
        "episode_latency": episode_latencies,
        "episode_energy": episode_energies
    })
    results_df.to_csv("results/test_summary.csv", index=False)
    results_df.to_excel("results/test_summary.xlsx", index=False, engine="openpyxl")


if __name__ == "__main__":
    run_test(args)

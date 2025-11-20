"""
Single-file PyTorch training script inspired by InforMARL:
"Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation"
(see: https://arxiv.org/pdf/2211.02127) and the official implementation:
https://github.com/nsidn98/InforMARL.
Usage (example):
     python algos/inforMARL.py --env_module sysadmin_network --total_steps 200000
Notes / citations:
 - Implementational choices were guided by the InforMARL paper and repository.
   See: https://arxiv.org/pdf/2211.02127 and https://github.com/nsidn98/InforMARL. :contentReference[oaicite:1]{index=1}

Part of the code was generated using LLMs.
"""

import argparse
import time
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

from cognac.utils.make_env import make_env

# -------------------------
# Hyperparameters / config
# -------------------------
DEFAULT_CFG = dict(
    seed=0,
    device="cpu",
    lr=5e-4,
    critic_lr=5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    ppo_epochs=10,
    minibatch_size=64,
    rollout_steps=4096,
    total_steps=2000000,
    hidden_dim=64,
    value_hidden=64,
    max_grad_norm=10.0,
    ent_coef=0.01,
    vf_coef=0.5,
    normalize_adv=True,
    encode_multidiscrete=True,
    max_onehot_dim=512,
)


# -------------------------
# Utilities & Networks
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(in_dim, hidden_dims, out_dim=None, activation=nn.ReLU, out_activation=None):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
        if out_activation is not None:
            layers.append(out_activation())
    return nn.Sequential(*layers)


class ObsPreprocessor:
    def __init__(self, example_obs, cfg):
        self.cfg = cfg
        self.obs_space = None
        self._setup_from_obs(example_obs)

    def _setup_from_obs(self, obs):
        if isinstance(obs, np.ndarray):
            self.obs_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.shape, dtype=obs.dtype
            )
        if hasattr(obs, "dtype") and np.issubdtype(obs.dtype, np.integer):
            self.try_onehot = True
        else:
            self.try_onehot = False

    def obs_to_vector(self, obs):
        if (
            isinstance(obs, np.ndarray)
            and np.issubdtype(obs.dtype, np.integer)
            and self.cfg["encode_multidiscrete"]
        ):
            if obs.ndim == 0:
                return np.array([float(obs)], dtype=np.float32)
            maxv = int(obs.max()) if obs.size > 0 else 0
            if maxv < 50 and obs.size * (maxv + 1) <= self.cfg["max_onehot_dim"]:
                parts = []
                for v in obs.flatten():
                    vec = np.zeros(maxv + 1, dtype=np.float32)
                    vec[int(v)] = 1.0
                    parts.append(vec)
                return np.concatenate(parts).astype(np.float32)
        try:
            return obs.astype(np.float32).flatten()
        except Exception:
            return np.array([float(obs)], dtype=np.float32)


class InfoAggregatorAttention(nn.Module):
    """
    Implementation of the Attention-based aggregation (UniMP style)
    described in InforMARL[cite: 149, 153].
    """

    def __init__(self, node_input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert (
            self.head_dim * num_heads == hidden_dim
        ), "Hidden dim must be divisible by num_heads"

        # Encoders
        self.input_mlp = mlp(node_input_dim, [hidden_dim], out_dim=hidden_dim)

        # Attention Weights (Query, Key, Value)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feats, adj):
        # node_feats: (B, N, D)
        # adj: (B, N, N) - treated as a mask (1 if connected, 0 if not)

        B, N, _ = node_feats.shape
        h = self.input_mlp(node_feats)  # (B, N, H)

        # Calculate Q, K, V
        Q = (
            self.W_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )  # (B, Heads, N, HeadDim)
        K = self.W_k(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # (B, Heads, N, HeadDim) @ (B, Heads, HeadDim, N) -> (B, Heads, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply Adjacency Mask (InforMARL restricts attention to local neighborhood [cite: 150])
        # We set scores to -inf where adj is 0 so softmax results in 0 attention.
        mask = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, Heads, N, N)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, Heads, N, N)

        # Aggregate
        # (B, Heads, N, N) @ (B, Heads, N, HeadDim) -> (B, Heads, N, HeadDim)
        out = torch.matmul(attn_weights, V)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)

        # Final projection + Residual connection (Standard in Graph Transformers)
        out = self.W_o(out) + h

        return out


class InfoAggregatorGraph(nn.Module):
    """
    GNN Aggregator.
    Accepts (Batch, N, Dim) inputs and (Batch, N, N) Adjacency.
    """

    def __init__(self, node_input_dim, hidden_dim, K=2, normalize=True):
        super().__init__()
        self.K = K
        self.normalize = normalize
        self.input_mlp = mlp(node_input_dim, [hidden_dim], out_dim=hidden_dim)
        self.update_mlp = mlp(hidden_dim, [hidden_dim], out_dim=hidden_dim)

    def forward(self, node_feats, adj):
        # node_feats: (B, N, D)
        # adj: (B, N, N)
        B, N, _ = node_feats.shape
        h = self.input_mlp(node_feats)

        # Normalize Adjacency
        adj_t = adj.float()
        if self.normalize:
            # Add self loops
            adj_t = adj_t + torch.eye(N, device=node_feats.device).unsqueeze(0)
            deg = adj_t.sum(dim=-1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
            deg_mat_left = deg_inv_sqrt.unsqueeze(-1)
            deg_mat_right = deg_inv_sqrt.unsqueeze(1)
            adj_norm = deg_mat_left * adj_t * deg_mat_right
        else:
            adj_norm = adj_t

        # Message Passing
        for _ in range(self.K):
            # (B,N,N) x (B,N,H) -> (B,N,H)
            m = torch.bmm(adj_norm, h)
            h = self.update_mlp(m)
        return h


class Actor(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = mlp(in_dim, [hidden_dim, hidden_dim], out_dim=action_dim)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = mlp(in_dim, [hidden_dim, hidden_dim], out_dim=1)

    def forward(self, x):
        return self.net(x)  # Output (Batch, 1)


# -------------------------
# Tensorized Rollout Buffer
# -------------------------
class RolloutBuffer:
    def __init__(self, num_agents, obs_dim, rollout_steps, device="cpu"):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.rollout_steps = rollout_steps
        self.device = device

        # Pre-allocate tensors: (Time, Agents, Dim)
        self.obs = torch.zeros((rollout_steps, num_agents, obs_dim), device=device)
        self.actions = torch.zeros((rollout_steps, num_agents), device=device)
        self.logps = torch.zeros((rollout_steps, num_agents), device=device)
        self.rewards = torch.zeros((rollout_steps, num_agents), device=device)
        self.dones = torch.zeros((rollout_steps, num_agents), device=device)
        self.values = torch.zeros((rollout_steps, num_agents), device=device)
        self.step = 0

    def add(self, obs, actions, logps, rewards, dones, values):
        # Expects inputs to be numpy arrays or tensors of shape (Num_Agents, ...)
        t = self.step
        self.obs[t] = torch.as_tensor(obs, device=self.device)
        self.actions[t] = torch.as_tensor(actions, device=self.device)
        self.logps[t] = torch.as_tensor(logps, device=self.device)
        self.rewards[t] = torch.as_tensor(rewards, device=self.device)
        self.dones[t] = torch.as_tensor(dones, device=self.device)
        self.values[t] = torch.as_tensor(values, device=self.device)
        self.step += 1

    def finish_rollout(self, last_values, gamma, lam):
        # Calculate GAE vectorially
        # last_values: (Num_Agents,)
        rewards = self.rewards.cpu().numpy()
        values = self.values.cpu().numpy()
        dones = self.dones.cpu().numpy()
        last_values = last_values

        advantages = np.zeros_like(rewards)
        lastgaelam = np.zeros(self.num_agents)

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                nextnonterminal = (
                    1.0 - dones[t]
                )  # Note: depends on how you handle truncation
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return torch.tensor(
            advantages, device=self.device, dtype=torch.float32
        ), torch.tensor(returns, device=self.device, dtype=torch.float32)


# -------------------------
# Trainer
# -------------------------
class PPOTrainer:
    def __init__(self, env, cfg, writer):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.env = env
        self.writer = writer
        # Adjacency: Ensure it's on device and float
        self.adjacency_matrix = torch.tensor(
            self.env.adjacency_matrix, device=self.device
        ).float()

        obs, _ = self.env.reset()
        self.agents = list(obs.keys())
        self.num_agents = len(self.agents)

        self.preproc = ObsPreprocessor(next(iter(obs.values())), self.cfg)
        example_vec = self.preproc.obs_to_vector(next(iter(obs.values())))
        self.obs_dim = example_vec.shape[0]
        self.agg_dim = self.cfg["hidden_dim"]

        # Assuming Discrete for simplicity
        act_space = self.env.action_space(self.agents[0])
        self.action_dim = act_space.n

        self._build_models()
        print(
            f"[INFO] Agents: {self.num_agents}, Obs: {self.obs_dim}, Agg: {self.agg_dim}"
        )

    def _build_models(self):
        in_actor = self.obs_dim + self.agg_dim
        self.actor = Actor(in_actor, self.cfg["hidden_dim"], self.action_dim).to(
            self.device
        )
        self.critic = Critic(self.agg_dim, self.cfg["value_hidden"]).to(self.device)
        # self.agg = InfoAggregatorGraph(self.obs_dim, self.agg_dim).to(self.device)
        self.agg = InfoAggregatorAttention(self.obs_dim, self.agg_dim).to(self.device)

        # Single optimizer usually easier to manage for coupled networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.agg.parameters()),
            lr=self.cfg["lr"],
        )

    def get_action_and_value(self, obs_dict, train_mode=False):
        """
        Takes raw observation dict.
        Returns actions, logps, values, and the processed obs vector.
        """
        # 1. Preprocess Obs
        obs_list = [self.preproc.obs_to_vector(obs_dict[a]) for a in self.agents]
        obs_t = torch.tensor(
            np.stack(obs_list), device=self.device, dtype=torch.float32
        )  # (N, Obs)

        # 2. Aggregation (Graph Pass)
        # We need to add a batch dimension (1, N, Obs) and (1, N, N)
        batch_obs = obs_t.unsqueeze(0)
        batch_adj = self.adjacency_matrix.unsqueeze(0)

        if train_mode:
            # Keep gradients
            agg_out = self.agg(batch_obs, batch_adj)  # (1, N, Agg)
        else:
            with torch.no_grad():
                agg_out = self.agg(batch_obs, batch_adj)

        agg_per_agent = agg_out.squeeze(0)  # (N, Agg)
        global_agg = agg_per_agent.mean(dim=0, keepdim=True)  # (1, Agg)

        # 3. Actor
        actor_inp = torch.cat([obs_t, agg_per_agent], dim=-1)  # (N, Obs+Agg)
        logits = self.actor(actor_inp)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logps = dist.log_prob(actions)

        # 4. Critic
        # Note: The original logic was centralized value (same value for all agents)
        # We calculate one value, then broadcast it to N agents for storage
        value_pred = self.critic(global_agg)  # (1, 1)
        values = value_pred.view(-1).expand(self.num_agents)  # (N,)

        return actions, logps, values, obs_t

    def train(self):
        cfg = self.cfg
        total_steps = cfg["total_steps"]
        rollout_steps = cfg["rollout_steps"]
        step_count = 0
        ep_rews = deque(maxlen=100)

        # Init Buffer
        buffer = RolloutBuffer(
            self.num_agents, self.obs_dim, rollout_steps, self.device
        )
        obs, _ = self.env.reset()
        ep_rew = 0

        while step_count < total_steps:
            buffer.step = 0  # Reset buffer pointer

            # --- ROLLOUT PHASE ---
            for t in range(rollout_steps):
                with torch.no_grad():
                    # Get actions (no grad needed here, just data collection)
                    actions_t, logps_t, values_t, obs_vec_t = self.get_action_and_value(
                        obs, train_mode=False
                    )

                # Convert to numpy for Env
                act_dict = {a: actions_t[i].item() for i, a in enumerate(self.agents)}

                next_obs, rewards, dones, trunc, _ = self.env.step(act_dict)
                # Store in buffer
                rew_vec = np.array([rewards[a] for a in self.agents])
                done_vec = np.array([dones[a] for a in self.agents])

                buffer.add(
                    obs_vec_t.cpu().numpy(),
                    actions_t.cpu().numpy(),
                    logps_t.cpu().numpy(),
                    rew_vec,
                    done_vec,
                    values_t.cpu().numpy(),
                )

                ep_rew += rew_vec.mean()
                obs = next_obs
                step_count += 1

                if all(dones.values()):
                    obs, _ = self.env.reset()
                    ep_rews.append(ep_rew)
                    ep_rew = 0

            # --- BOOTSTRAP ---
            with torch.no_grad():
                _, _, last_vals, _ = self.get_action_and_value(obs, train_mode=False)

            advantages, returns = buffer.finish_rollout(
                last_vals.cpu().numpy(), cfg["gamma"], cfg["gae_lambda"]
            )

            # --- PPO UPDATE PHASE ---
            # Flattened: (Steps * Agents, ...) is standard, BUT we need (Batch, Agents, ...) to run GNN
            # Strategy: Batch over TIME Steps.

            b_obs = buffer.obs  # (Steps, N, D)
            b_act = buffer.actions  # (Steps, N)
            b_logp = buffer.logps  # (Steps, N)
            b_adv = advantages  # (Steps, N)
            b_ret = returns  # (Steps, N)

            if cfg["normalize_adv"]:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            num_timesteps = b_obs.shape[0]  # rollout_steps
            batch_size_time = cfg[
                "minibatch_size"
            ]  # e.g., 32 time steps (graph snapshots)

            for _ in range(cfg["ppo_epochs"]):
                indices = torch.randperm(num_timesteps)

                for start in range(0, num_timesteps, batch_size_time):
                    idx = indices[start : start + batch_size_time]

                    # 1. Slice Batch over Time
                    mb_obs = b_obs[idx]  # (B, N, D)
                    mb_act = b_act[idx]  # (B, N)
                    mb_old_logp = b_logp[idx]  # (B, N)
                    mb_adv = b_adv[idx]  # (B, N)
                    mb_ret = b_ret[idx]  # (B, N)

                    cur_batch_size = mb_obs.size(0)

                    # 2. RECOMPUTE GNN AGGREGATION
                    # Expand Adj to match batch size: (B, N, N)
                    mb_adj = self.adjacency_matrix.unsqueeze(0).expand(
                        cur_batch_size, -1, -1
                    )

                    # Forward GNN - Gradient flows here!
                    mb_agg = self.agg(mb_obs, mb_adj)  # (B, N, AggDim)

                    # 3. Prepare Actor Inputs
                    # We now Flatten B and N for the MLP part
                    # mb_obs: (B, N, D) -> (B*N, D)
                    # mb_agg: (B, N, H) -> (B*N, H)
                    flat_obs = mb_obs.view(-1, self.obs_dim)
                    flat_agg = mb_agg.view(-1, self.agg_dim)

                    actor_inp = torch.cat([flat_obs, flat_agg], dim=-1)

                    # 4. Actor Loss
                    logits = self.actor(actor_inp)
                    dist = torch.distributions.Categorical(logits=logits)

                    flat_act = mb_act.view(-1)
                    new_logp = dist.log_prob(flat_act)
                    entropy = dist.entropy().mean()

                    flat_old_logp = mb_old_logp.view(-1)
                    flat_adv = mb_adv.view(-1)

                    ratio = (new_logp - flat_old_logp).exp()
                    surr1 = ratio * flat_adv
                    surr2 = (
                        torch.clamp(
                            ratio, 1.0 - cfg["clip_epsilon"], 1.0 + cfg["clip_epsilon"]
                        )
                        * flat_adv
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # 5. Critic Loss
                    # Critic takes global mean of aggregation: (B, N, H) -> (B, H)
                    mb_global_agg = mb_agg.mean(dim=1)
                    value_pred = self.critic(mb_global_agg)  # (B, 1)

                    # We need to match returns shape. Returns are (B, N).
                    # We expand value_pred to (B, N) effectively saying "global value applies to all agents"
                    value_pred_expanded = value_pred.view(-1, 1).expand(
                        -1, self.num_agents
                    )

                    # Flatten for MSE
                    flat_ret = mb_ret.view(-1)
                    flat_val = value_pred_expanded.reshape(-1)

                    value_loss = 0.5 * (flat_val - flat_ret).pow(2).mean()

                    loss = (
                        policy_loss
                        + cfg["vf_coef"] * value_loss
                        - cfg["ent_coef"] * entropy
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), cfg["max_grad_norm"]
                    )
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), cfg["max_grad_norm"]
                    )
                    nn.utils.clip_grad_norm_(
                        self.agg.parameters(), cfg["max_grad_norm"]
                    )
                    self.optimizer.step()

            self.writer.add_scalar(
                "training/ep_reward", np.mean(ep_rews) if ep_rews else 0, step_count
            )
            if step_count % 1000 < rollout_steps:
                print(
                    f"[STEP {step_count}] Avg Reward: {np.mean(ep_rews) if ep_rews else 0:.2f}"
                )

        print("Training finished.")


# -------------------------
# Main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_module", type=str, default="sysadmin_network"
    )  # Example default
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=50000)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = DEFAULT_CFG.copy()
    cfg.update({"hidden_dim": args.hidden_dim, "total_steps": args.total_steps})

    # Mocking loading the adjacency for the example
    # In real use, ensure adjacency matches env.
    # Here we assume the env provides it or we load a dummy.
    adjacency_matrix = np.load("env_assets/basic_directed_network_10.npy")

    writer = SummaryWriter(f"runs/InforMARL_Corrected_{time.time()}")
    env = make_env(
        args.env_module,
        **{"adjacency_matrix": adjacency_matrix, "is_shared_reward": True},
    )

    trainer = PPOTrainer(env, cfg, writer)
    trainer.train()


if __name__ == "__main__":
    main()

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

Part of the code was generated using ChatGPT.
"""

import argparse
import importlib
import math
import time
import os
import random
from collections import deque, defaultdict, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

# PettingZoo imports; convert AEC -> Parallel if necessary
try:
    from pettingzoo.utils.conversions import aec_to_parallel
    from pettingzoo.utils import parallel_to_aec
    from pettingzoo import ParallelEnv
except Exception as e:
    raise ImportError(
        "Please install pettingzoo (pip install 'pettingzoo') to use this script."
    ) from e
from cognac.utils.make_env import make_env

# -------------------------
# Hyperparameters / config
# -------------------------
DEFAULT_CFG = dict(
    seed=0,
    device="cpu",  # CPU by request
    lr=3e-4,
    critic_lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    ppo_epochs=4,
    minibatch_size=64,
    rollout_steps=128,  # steps per env per update
    total_steps=200000,
    num_envs=1,  # how many parallel env instances (if user wraps multiple envs)
    hidden_dim=128,
    value_hidden=128,
    max_grad_norm=0.5,
    ent_coef=0.01,
    vf_coef=0.5,
    normalize_adv=True,
    use_shared_policy=True,  # parameter-sharing across agents
    use_value_norm=False,
    encode_multidiscrete=True,  # attempt to one-hot MultiDiscrete obs
    max_onehot_dim=512,  # if a single multi-discrete dimension > this, we won't one-hot encode
)


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def try_import_env(env_module_spec: str):
    """
    env_module_spec: "module:fn" string where fn() returns a PettingZoo env (AEC or Parallel).
    """
    if ":" not in env_module_spec:
        raise ValueError("env_module must be of form module:fn (e.g. myenv:make_env)")
    module_name, fn_name = env_module_spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    env = fn()
    return env


# -------------------------
# Observation preprocessing
# -------------------------
class ObsPreprocessor:
    """
    Handles conversion of PettingZoo observations to flat tensors for networks.
    MultiDiscrete --> one-hot across each sub-dim unless prohibitive.
    Otherwise falls back to flattening and casting to float.
    """

    def __init__(self, example_obs, cfg):
        self.cfg = cfg
        self.obs_space = None
        self._setup_from_obs(example_obs)

    def _setup_from_obs(self, obs):
        # Try to infer gym.Space if possible; obs may be numpy arrays
        # For MultiDiscrete, we store nvec
        if isinstance(obs, np.ndarray):
            self.obs_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.shape, dtype=obs.dtype
            )
        else:
            # best effort: if user provided a gym Space object itself
            self.obs_space = None

        # If obs is a dict / tuple, we fall back to flatten.
        # If MultiDiscrete array-like is detected, attempt one-hot.
        if hasattr(obs, "dtype") and np.issubdtype(obs.dtype, np.integer):
            # treat as possible MultiDiscrete-like int vector
            # We will attempt to one-hot encode each index if cfg allows.
            self.try_onehot = True
        else:
            self.try_onehot = False

        # We won't know sizes per dimension in generic case; decide at runtime per agent-observation
        # No permanent representation is constructed here.

    def obs_to_vector(self, obs):
        # obs: numpy array or scalar or nested
        if (
            isinstance(obs, np.ndarray)
            and np.issubdtype(obs.dtype, np.integer)
            and self.cfg["encode_multidiscrete"]
        ):
            # attempt per-dim one-hot if dims small
            if obs.ndim == 0:
                return np.array([float(obs)], dtype=np.float32)
            parts = []
            # assume each integer in obs is an index with unknown upper bound; we cannot reliably one-hot without knowing nvec
            # so fallback to scaling indices into floats if onehot would be large or unknown.
            # Heuristic: if max(obs) < 50 and len(obs)* (max(obs)+1) < max_onehot_dim -> one-hot
            maxv = int(obs.max()) if obs.size > 0 else 0
            if maxv < 50 and obs.size * (maxv + 1) <= self.cfg["max_onehot_dim"]:
                for v in obs.flatten():
                    vec = np.zeros(maxv + 1, dtype=np.float32)
                    vec[int(v)] = 1.0
                    parts.append(vec)
                return np.concatenate(parts).astype(np.float32)
            else:
                return obs.astype(np.float32).flatten()
        else:
            # numeric or float array
            try:
                return obs.astype(np.float32).flatten()
            except Exception:
                return np.array([float(obs)], dtype=np.float32)


# -------------------------
# Networks
# -------------------------
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


class InfoAggregator(nn.Module):
    """
    Simple information aggregation module.
    For each agent i, we take its local embedding e_i and compute an aggregated
    neighbor embedding x_i_agg by:
      - optionally applying a small MLP to each e_j
      - mean-pooling across all agents (or a neighbor subset, if provided)
      - optionally passing pooled vector through another MLP
    This is a compact, GNN-like mean aggregator usable when the env does not provide graph edges.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.node_mlp = mlp(input_dim, [hidden_dim], hidden_dim)
        self.post_mlp = mlp(hidden_dim, [hidden_dim], hidden_dim)

    def forward(self, node_embeddings: torch.Tensor):
        # node_embeddings: (num_agents, batch, emb_dim) or (num_agents, emb_dim)
        # We'll support (batch, num_agents, emb_dim)
        if node_embeddings.dim() == 3:
            # (batch, N, D)
            h = self.node_mlp(node_embeddings)  # maps per-node
            pooled = h.mean(dim=1)  # (batch, hidden_dim)
            out = self.post_mlp(pooled)  # (batch, hidden_dim)
            # For per-agent aggregated vector, tile back
            tiled = out.unsqueeze(1).expand(
                -1, node_embeddings.size(1), -1
            )  # (batch, N, hidden_dim)
            return tiled
        elif node_embeddings.dim() == 2:
            # (N, D) -> (N, H) then mean -> produce per-node tiled outcome
            h = self.node_mlp(node_embeddings)  # (N, H)
            pooled = h.mean(dim=0, keepdim=True)  # (1, H)
            out = self.post_mlp(pooled)  # (1, H)
            tiled = out.expand(node_embeddings.size(0), -1)  # (N, H)
            return tiled
        else:
            raise ValueError("Unsupported node_embeddings shape")


class InfoAggregatorGraph(nn.Module):
    """
    K-step message-passing aggregator using adjacency matrix.
    - node_input_dim: per-agent embedding dim (e.g., flattened obs dim)
    - hidden_dim: hidden node embedding dim
    - K: number of message-passing rounds
    - use_edge_weights: if True, adjacency values are treated as weights (floats)
    - normalize: if True, use symmetric normalization D^{-1/2} A D^{-1/2} before aggregation
    """

    def __init__(
        self, node_input_dim, hidden_dim, K=2, use_edge_weights=True, normalize=True
    ):
        super().__init__()
        self.K = K
        self.use_edge_weights = use_edge_weights
        self.normalize = normalize

        # initial node embedder
        self.input_mlp = mlp(node_input_dim, [hidden_dim], out_dim=hidden_dim)
        # message MLP applied after aggregation
        self.update_mlp = mlp(hidden_dim, [hidden_dim], out_dim=hidden_dim)

    def forward(self, node_feats, adj=None):
        """
        node_feats: torch.Tensor shape (batch, N, D)  OR (N, D) if no batch
        adj: None or torch.Tensor shape (batch, N, N) or (N, N) with 0/1 or weights
        returns: per-agent aggregated embeddings, shape (batch, N, hidden_dim) or (N, hidden_dim)
        """
        batched = node_feats.dim() == 3
        if not batched:
            node_feats = node_feats.unsqueeze(0)  # (1, N, D)
            if adj is not None and adj.dim() == 2:
                adj = adj.unsqueeze(0)

        B, N, _ = node_feats.shape
        h = self.input_mlp(node_feats)  # (B, N, H)

        if adj is None:
            # fallback to global mean pooling repeated per node (like before)
            pooled = h.mean(dim=1, keepdim=True)  # (B, 1, H)
            out = self.update_mlp(pooled).expand(-1, N, -1)  # (B, N, H)
            return out.squeeze(0) if not batched else out

        # Ensure adj is float tensor on same device
        adj_t = adj.float().to(h.device)  # (B, N, N)
        # Optionally normalize adjacency (symmetric)
        if self.normalize:
            # add self-loops for stability
            adj_t = adj_t + torch.eye(N, device=h.device).unsqueeze(0)
            deg = adj_t.sum(dim=-1)  # (B, N)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
            # D^{-1/2} A D^{-1/2}
            deg_mat_left = deg_inv_sqrt.unsqueeze(-1)  # (B, N, 1)
            deg_mat_right = deg_inv_sqrt.unsqueeze(1)  # (B, 1, N)
            adj_norm = deg_mat_left * adj_t * deg_mat_right  # broadcast -> (B, N, N)
        else:
            adj_norm = adj_t

        # Message passing rounds
        for _ in range(self.K):
            # aggregate neighbor messages: (B, N, N) @ (B, N, H) -> (B, N, H)
            m = torch.bmm(adj_norm, h)
            # combine with current representation (here just feed aggregated msg through MLP)
            h = self.update_mlp(m)

        # output per-agent embeddings
        return h.squeeze(0) if not batched else h


class Actor(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = mlp(in_dim, [hidden_dim, hidden_dim], out_dim=action_dim)

    def forward(self, x):
        # returns logits
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = mlp(in_dim, [hidden_dim, hidden_dim], out_dim=1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -------------------------
# Rollout Buffer
# -------------------------
RolloutItem = namedtuple(
    "RolloutItem", ["obs", "agg", "action", "logp", "reward", "done", "value"]
)


class RolloutBuffer:
    def __init__(self, num_agents, obs_dim, agg_dim, rollout_steps, device="cpu"):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.agg_dim = agg_dim
        self.rollout_steps = rollout_steps
        self.device = device
        self.reset()

    def reset(self):
        self.items = [[] for _ in range(self.num_agents)]

    def add(self, agent_idx, obs, agg, action, logp, reward, done, value):
        self.items[agent_idx].append(
            RolloutItem(obs, agg, action, logp, reward, done, value)
        )

    def compute_returns_and_advantages(self, last_values, gamma, lam):
        """
        last_values: list/np array of value estimates for each agent at final step (num_agents,)
        Returns flattened lists for learning.
        """
        per_agent_data = []
        for i in range(self.num_agents):
            rews = [it.reward for it in self.items[i]]
            vals = [it.value for it in self.items[i]]
            dones = [it.done for it in self.items[i]]
            T = len(rews)
            advs = np.zeros(T, dtype=np.float32)
            lastgaelam = 0
            for t in reversed(range(T)):
                nonterm = 1.0 - float(dones[t])
                next_val = last_values[i] if t == T - 1 else vals[t + 1]
                delta = rews[t] + gamma * next_val * nonterm - vals[t]
                lastgaelam = delta + gamma * lam * nonterm * lastgaelam
                advs[t] = lastgaelam
            returns = advs + np.array(vals, dtype=np.float32)
            per_agent_data.append((self.items[i], advs, returns))
        return per_agent_data


# -------------------------
# Trainer
# -------------------------
class PPOTrainer:
    def __init__(self, env, cfg, writer):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.env = env
        self.writer = writer
        self.adjacency_matrix = torch.tensor(self.env.adjacency_matrix).float()
        self.agents = None  # list of agent ids in env
        self.is_parallel = True
        self._init_env_and_spaces()
        self._build_models()

    def _init_env_and_spaces(self):
        # Reset to get agent list and example observations
        obs, _ = self.env.reset()
        # obs is a dict mapping agent_id -> observation
        self.agents = list(obs.keys())
        self.num_agents = len(self.agents)
        # example observation for preprocessor
        self.preproc = ObsPreprocessor(next(iter(obs.values())), self.cfg)
        # compute observation vector sizes by converting an example obs
        example_vec = self.preproc.obs_to_vector(next(iter(obs.values())))
        self.obs_dim = example_vec.shape[0]
        # aggregator hidden dim
        self.agg_dim = self.cfg["hidden_dim"]
        # action spaces: assume discrete for all agents and same for simplicity; else we will handle heterogeneous via dicts
        example_agent = self.agents[0]
        act_space = self.env.action_space(example_agent)
        if isinstance(act_space, spaces.Discrete):
            self.action_dim = act_space.n
            self.discrete = True
        else:
            # If not Discrete, try to handle it generically (but user specified Discrete)
            if hasattr(act_space, "nvec"):
                # MultiDiscrete action -> flatten into choices by product? not handled here.
                raise NotImplementedError(
                    "MultiDiscrete action spaces are not implemented by this simple trainer."
                )
            else:
                raise NotImplementedError(
                    "Only Discrete action spaces are supported in this script."
                )
        print(
            f"[INFO] Detected {self.num_agents} agents. obs_dim={self.obs_dim} action_dim={self.action_dim}"
        )

    def _build_models(self):
        in_actor = self.obs_dim + self.agg_dim
        self.actor = Actor(in_actor, self.cfg["hidden_dim"], self.action_dim).to(
            self.device
        )
        # Parameter sharing: single actor used for all agents
        self.critic = Critic(self.agg_dim, self.cfg["value_hidden"]).to(self.device)
        self.agg = InfoAggregatorGraph(self.obs_dim, self.agg_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg["lr"])
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.cfg["critic_lr"]
        )

    def select_actions(self, obs_dict):
        """
        obs_dict: agent_id -> observation (numpy)
        returns: actions (dict), logps (dict), values (dict)
        """
        # Build batch of agent obs vectors
        obs_list = []
        for a in self.agents:
            vec = self.preproc.obs_to_vector(obs_dict[a])
            obs_list.append(vec)
        # (N, obs_dim)
        obs_np = np.stack(obs_list, axis=0).astype(np.float32)
        obs_t = torch.tensor(obs_np, device=self.device)
        # For aggregator we expect (batch, N, D). Here batch=1
        node_emb = obs_t.unsqueeze(0)  # (1, N, D)
        agg_per_agent = self.agg(node_emb, adj=self.adjacency_matrix)  # (1, N, H)
        agg_per_agent = agg_per_agent.squeeze(0)  # (N, H)
        # For critic we compute global aggregated vector (mean across agents)
        global_agg = agg_per_agent.mean(dim=0, keepdim=False)  # (H,)
        # compute actions per agent
        actions = {}
        logps = {}
        values = {}
        for idx, a in enumerate(self.agents):
            ob = obs_t[idx]
            ag = agg_per_agent[idx]
            inp = torch.cat([ob, ag], dim=-1).unsqueeze(0)  # (1, in_actor)
            logits = self.actor(inp)  # (1, action_dim)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)
            # critic uses global aggregated vector
            val = self.critic(global_agg.unsqueeze(0))
            actions[a] = int(act.item())
            logps[a] = float(logp.item())
            values[a] = float(val.item())
        return (
            actions,
            logps,
            values,
            obs_np,
            agg_per_agent.detach().cpu().numpy(),
            float(global_agg.detach().cpu().numpy().mean()),
        )

    def train(self):
        cfg = self.cfg
        total_steps = cfg["total_steps"]
        rollout_steps = cfg["rollout_steps"]
        step = 0
        episode = 0
        ep_rews = deque(maxlen=100)

        while step < total_steps:
            # Collect rollout
            buffer = RolloutBuffer(
                self.num_agents,
                self.obs_dim,
                self.agg_dim,
                rollout_steps,
                device=self.device,
            )
            obs, _ = self.env.reset()
            ep_rew = 0
            # For rollouts across time
            for t in range(rollout_steps):
                actions, logps, values, obs_np, per_agent_agg, global_agg_mean = (
                    self.select_actions(obs)
                )
                next_obs, rewards, dones, truncations, infos = self.env.step(actions)
                ep_rew += sum(list(rewards.values()))
                # record for each agent
                for i, a in enumerate(self.agents):
                    buffer.add(
                        i,
                        obs_np[i],
                        per_agent_agg[i],
                        actions[a],
                        logps[a],
                        float(rewards[a]),
                        bool(dones[a]),
                        values[a],
                    )
                obs = next_obs
                step += 1
                # quick episode logging if terminated
                if all(dones.values()):
                    episode += 1
                    # estimate episode reward from rewards in last step (rough)
                    ep_rews.append(ep_rew)
                    break
            # compute last values for bootstrap
            # get value estimates at last obs
            _, _, last_values, _, _, _ = self.select_actions(obs)
            # last_values is dict agent->value
            last_vals_arr = [last_values[a] for a in self.agents]

            per_agent_data = buffer.compute_returns_and_advantages(
                last_vals_arr, cfg["gamma"], cfg["gae_lambda"]
            )

            # flatten data across agents to form batches for PPO update
            obs_batch = []
            agg_batch = []
            act_batch = []
            oldlogp_batch = []
            adv_batch = []
            ret_batch = []
            for agent_idx, (items, advs, returns) in enumerate(per_agent_data):
                for j, it in enumerate(items):
                    obs_batch.append(it.obs)
                    agg_batch.append(it.agg)
                    act_batch.append(it.action)
                    oldlogp_batch.append(it.logp)
                    adv_batch.append(advs[j])
                    ret_batch.append(returns[j])

            obs_batch = torch.tensor(
                np.stack(obs_batch, axis=0), dtype=torch.float32, device=self.device
            )
            agg_batch = torch.tensor(
                np.stack(agg_batch, axis=0), dtype=torch.float32, device=self.device
            )
            act_batch = torch.tensor(
                np.array(act_batch), dtype=torch.int64, device=self.device
            )
            oldlogp_batch = torch.tensor(
                np.array(oldlogp_batch), dtype=torch.float32, device=self.device
            )
            adv_batch = torch.tensor(
                np.array(adv_batch), dtype=torch.float32, device=self.device
            )
            ret_batch = torch.tensor(
                np.array(ret_batch), dtype=torch.float32, device=self.device
            )

            if cfg["normalize_adv"]:
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

            # create input to actor: [obs, agg]
            actor_inp = torch.cat([obs_batch, agg_batch], dim=-1)

            # PPO epochs
            dataset_size = actor_inp.size(0)
            for _epoch in range(cfg["ppo_epochs"]):
                # minibatch sampling
                perm = torch.randperm(dataset_size)
                for start in range(0, dataset_size, cfg["minibatch_size"]):
                    idx = perm[start : start + cfg["minibatch_size"]]
                    batch_x = actor_inp[idx]
                    batch_actions = act_batch[idx]
                    batch_oldlogp = oldlogp_batch[idx]
                    batch_adv = adv_batch[idx]
                    batch_ret = ret_batch[idx]

                    logits = self.actor(batch_x)
                    dist = torch.distributions.Categorical(logits=logits)
                    newlogp = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    ratio = (newlogp - batch_oldlogp).exp()
                    surr1 = ratio * batch_adv
                    surr2 = (
                        torch.clamp(
                            ratio, 1.0 - cfg["clip_epsilon"], 1.0 + cfg["clip_epsilon"]
                        )
                        * batch_adv
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # critic uses aggregated vectors only: we need to compute critic input for the corresponding samples.
                    # Here batch_ret are returns; we need to compute value predictions for these samples.
                    # Our critic expects aggregated vector (agg_dim) input; extract from agg_batch
                    critic_inp = agg_batch[idx]
                    value_preds = self.critic(critic_inp).squeeze(-1)
                    value_loss = (value_preds - batch_ret).pow(2).mean()

                    loss = (
                        policy_loss
                        + cfg["vf_coef"] * value_loss
                        - cfg["ent_coef"] * entropy
                    )

                    self.optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), cfg["max_grad_norm"]
                    )
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), cfg["max_grad_norm"]
                    )
                    self.optimizer.step()
                    self.critic_optimizer.step()

            # end update
            self.writer.add_scalar(
                "global_measure/episodic_return",
                np.mean(ep_rews),
                step,
            )
            print(
                f"[STEP {step}] Update complete. avg_ep_reward (recent) = {np.mean(ep_rews)}"
            )
        print("Training finished.")


# -------------------------
# Argument parsing & run
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_module",
        type=str,
        required=True,
        help="Python module:callable that returns a PettingZoo env, e.g. myenv:make_env",
    )
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--total_steps", type=int, default=DEFAULT_CFG["total_steps"])
    parser.add_argument(
        "--rollout_steps", type=int, default=DEFAULT_CFG["rollout_steps"]
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CFG["hidden_dim"])
    parser.add_argument(
        "--minibatch_size", type=int, default=DEFAULT_CFG["minibatch_size"]
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = DEFAULT_CFG.copy()
    cfg.update(
        {
            "hidden_dim": args.hidden_dim,
            "minibatch_size": args.minibatch_size,
            "rollout_steps": args.rollout_steps,
            "total_steps": args.total_steps,
            "seed": args.seed,
        }
    )
    set_seed(cfg["seed"])
    ADJ = "basic_directed_network_1000.npy"
    run_name = f"InforMARL_shared_{ADJ}_{time.time()}"
    writer = SummaryWriter(f"runs/{run_name}")
    adjacency_matrix_path = os.path.join(os.path.dirname(__file__), "env_assets", ADJ)
    adjacency_matrix = np.load(adjacency_matrix_path)
    env = make_env(args.env_module, **{"adjacency_matrix": adjacency_matrix})
    _, _ = env.reset()  # Instantiate agents
    trainer = PPOTrainer(env, cfg, writer)
    trainer.train()


if __name__ == "__main__":
    main()

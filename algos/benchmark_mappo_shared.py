"""
MAPPO-like training with a shared policy across homogeneous agents (single actor for all agents)
and a centralized critic.
"""

import time
from collections import deque, defaultdict, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter

from cognac.utils.make_env import make_env


# ---------------- Utilities -----------------
def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def flatten_obs(obs):
    # if obs is already flat numeric array, return as np.array
    return np.asarray(obs, dtype=np.float32).reshape(-1)


# Simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Actor that supports discrete or continuous actions
class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_space, hidden=(256, 256)):
        super().__init__()
        self.is_discrete = hasattr(action_space, "n") and (
            getattr(action_space, "n", None) is not None
        )
        if self.is_discrete:
            self.logits_net = MLP(obs_dim, action_space.n, hidden=hidden)
        else:
            # continuous action: assume Box with shape (act_dim,)
            act_dim = int(np.prod(action_space.shape))
            self.mean_net = MLP(obs_dim, act_dim, hidden=hidden)
            # learnable log_std per action dim
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_tensor):
        # obs_tensor: (batch, obs_dim)
        if self.is_discrete:
            logits = self.logits_net(obs_tensor)  # (batch, n_actions)
            return logits
        else:
            mean = self.mean_net(obs_tensor)  # (batch, act_dim)
            std = torch.exp(self.log_std)
            return mean, std

    def get_action_and_logp(self, obs_tensor):
        if self.is_discrete:
            logits = self.forward(obs_tensor)
            dist = Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)
            return act.cpu().numpy(), logp, dist
        else:
            mean, std = self.forward(obs_tensor)
            dist = Normal(mean, std)
            act = dist.sample()
            # For continuous envs, some envs expect numpy arrays of shape (act_dim,)
            logp = dist.log_prob(act).sum(dim=-1)
            return act.cpu().numpy(), logp, dist


# Centralized value function
class CentralCritic(nn.Module):
    def __init__(self, state_dim, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(state_dim, 1, hidden=hidden)

    def forward(self, state_tensor):
        return self.net(state_tensor).squeeze(-1)  # (batch,)


# --------- Experience buffer for PPO / GAE -----------
Experience = namedtuple(
    "Experience", ["obs", "state", "action", "logp", "reward", "done", "value"]
)


class RolloutBuffer:
    def __init__(self):
        self.storage = []

    def add(self, **kwargs):
        self.storage.append(Experience(**kwargs))

    def clear(self):
        self.storage = []

    def get(self):
        return self.storage


# ---------- Training loop and helpers ----------
def compute_gae(buf, critic, gamma=0.99, lam=0.95, device="cpu"):
    exps = buf.get()
    # We'll compute returns and advantages per time-step using values from critic where needed.
    # exps is list of Experience with fields containing batch items (for all agents flattened).
    rewards = np.array([e.reward for e in exps], dtype=np.float32)
    dones = np.array([e.done for e in exps], dtype=np.float32)
    values = np.array([e.value for e in exps], dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    # reversed
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        nextval = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextval * nonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


def ppo_update(
    actor,
    critic,
    optimizer_actor,
    optimizer_critic,
    batch,
    advantages,
    returns,
    clip_eps=0.2,
    vf_coef=1.0,
    ent_coef=0.0,
    epochs=4,
    minibatch_size=64,
    device="cpu",
):
    """
    batch: dict keys -> numpy arrays: obs, state, actions, old_logp
    advantages, returns: numpy arrays (same length)
    """
    obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    state = torch.tensor(batch["state"], dtype=torch.float32, device=device)
    actions = torch.tensor(batch["actions"], device=device)
    old_logp = torch.tensor(batch["old_logp"], dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    N = obs.shape[0]

    for _ in range(epochs):
        # shuffle indices
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        for start in range(0, N, minibatch_size):
            mb_idx = idxs[start : start + minibatch_size]
            mb_obs = obs[mb_idx]
            mb_state = state[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logp = old_logp[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = returns[mb_idx]

            # Policy forward
            if actor.is_discrete:
                logits = actor(mb_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions.long())
                entropy = dist.entropy().mean()
            else:
                mean, std = actor(mb_obs)
                dist = Normal(mean, std)
                new_logp = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values_pred = critic(mb_state)
            value_loss = ((values_pred - mb_ret) ** 2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer_actor.step()
            optimizer_critic.step()


# ---------- Main training function ----------
def train_shared_mappo(
    env_factory,
    writer,
    total_timesteps=10_000_000,
    rollout_length=128,
    gamma=0.99,
    lam=0.95,
    lr_actor=3e-4,
    lr_critic=3e-4,
    clip_eps=0.2,
    vf_coef=1.0,
    ent_coef=0.0,
    ppo_epochs=4,
    minibatch_size=64,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = env_factory()
    agent_ids = list(env.agents)
    num_agents = len(agent_ids)
    print(f"[INFO] num_agents = {num_agents}, agents = {agent_ids}")

    # Inspect spaces using the first reset observation
    obs0, _ = env.reset()
    # flatten and determine obs dim per agent
    sample_obs = flatten_obs(obs0[agent_ids[0]])
    obs_dim = sample_obs.size if hasattr(sample_obs, "size") else sample_obs.shape[0]
    # determine action space type using first agent
    try:
        action_space = env.action_space(agent_ids[0])
    except Exception:
        # some envs have env.action_spaces mapping
        action_space = env.action_spaces[agent_ids[0]]

    # Determine centralized state dim (preferred)
    try:
        state0 = env.state()
        state_flat = np.asarray(state0, dtype=np.float32).reshape(-1)
        state_dim = state_flat.shape[0]
        use_state = True
    except Exception:
        # fallback: concatenated per-agent observations
        use_state = False
        # get obs for all agents concatenated
        concat_sample = np.concatenate(
            [flatten_obs(obs0[a]) for a in agent_ids], axis=0
        )
        state_dim = concat_sample.shape[0]

    print(
        f"[INFO] obs_dim (per agent) = {obs_dim}, state_dim = {state_dim}, use_state={use_state}"
    )

    actor = SharedActor(obs_dim, action_space).to(device)
    critic = CentralCritic(state_dim).to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    buffer = RolloutBuffer()
    timestep = 0
    episode_returns = deque(maxlen=50)
    ep_ret = 0
    ep_len = defaultdict(int)

    obs, _ = env.reset()
    start_time = time.time()

    while timestep < total_timesteps:
        # Collect rollout_length steps (each step contains actions for all agents)
        for _ in range(rollout_length):
            # Build batched obs for each agent in consistent order
            obs_batch = []
            for a in agent_ids:
                obs_batch.append(flatten_obs(obs[a]))
            obs_batch = np.stack(obs_batch, axis=0)  # shape (num_agents, obs_dim)

            obs_tensor = torch.tensor(
                obs_batch, dtype=torch.float32, device=device
            )  # (num_agents, obs_dim)

            # State for critic
            if use_state:
                state = env.state()
                state_flat = np.asarray(state, dtype=np.float32).reshape(1, -1)
                state_batch = np.repeat(state_flat, num_agents, axis=0)
            else:
                state_batch = np.concatenate(
                    [flatten_obs(obs[a]) for a in agent_ids], axis=0
                )
                # replicate full concatenated state for each agent (critic input is same for agent-level entries)
                state_batch = np.repeat(state_batch.reshape(1, -1), num_agents, axis=0)

            state_tensor = torch.tensor(state_batch, dtype=torch.float32, device=device)

            # actor -> actions & logps for all agents in parallel
            if actor.is_discrete:
                logits = actor(obs_tensor)
                dist = Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
                actions_for_env = {
                    agent_ids[i]: int(acts[i].cpu().item()) for i in range(num_agents)
                }
                # keep numpy / shape
                actions_np = acts.cpu().numpy()
                logps_np = logps.detach().cpu().numpy()
            else:
                mean, std = actor(obs_tensor)
                dist = Normal(mean, std)
                acts = dist.sample()
                logps = dist.log_prob(acts).sum(dim=-1)
                # convert actions to numpy with correct shape per agent
                actions_for_env = {
                    agent_ids[i]: acts[i].cpu().numpy() for i in range(num_agents)
                }
                actions_np = acts.detach().cpu().numpy()
                logps_np = logps.detach().cpu().numpy()

            # compute value for each agent (centralized critic: same input for every agent but we kept replicate)
            with torch.no_grad():
                values = (
                    critic(state_tensor).detach().cpu().numpy()
                )  # shape (num_agents,)

            # Step env
            next_obs, rewards, dones, trunc, infos = env.step(actions_for_env)
            # print(np.unique(list(actions_for_env.values()), return_counts=True)[1])
            # store experiences per agent as separate entries (flattened)
            for i, a in enumerate(agent_ids):
                buffer.add(
                    obs=obs_batch[i].astype(np.float32),
                    state=state_batch[i].astype(np.float32),
                    action=(
                        actions_np[i].astype(np.float32)
                        if not actor.is_discrete
                        else np.array(actions_np[i])
                    ),
                    logp=float(logps_np[i]),
                    reward=float(rewards.get(a, 0.0)),
                    done=bool(dones.get(a, False)),
                    value=float(values[i]),
                )
                # ep_ret[a] += float(rewards.get(a, 0.0))
                ep_len[a] += 1
                # if dones.get(a, False):
                #     episode_returns.append(ep_ret[a])
                #     print(episode_returns)
                #     # ep_ret[a] = 0.0
                #     # ep_len[a] = 0
            ep_ret += sum(list(rewards.values()))
            obs = next_obs
            timestep += 1  # counted as agent-steps (approx)
            if any(list(dones.values())):
                obs, _ = env.reset()
                # print(ep_ret)
                episode_returns.append(ep_ret)
                writer.add_scalar(
                    "global_measure/episodic_return",
                    ep_ret,
                    timestep,
                )
                ep_ret = 0
            if timestep >= total_timesteps:
                break

        # At end of rollout compute advantages+returns
        exps = buffer.get()
        if len(exps) == 0:
            continue
        advantages, returns = compute_gae(
            buffer, critic, gamma=gamma, lam=lam, device=device
        )

        # Build batch dict for PPO update
        # concatenate per-time-step entries into arrays
        obs_array = np.stack([e.obs for e in exps], axis=0)
        state_array = np.stack([e.state for e in exps], axis=0)
        actions_array = np.stack([e.action for e in exps], axis=0)
        old_logp_array = np.array([e.logp for e in exps], dtype=np.float32)

        batch = {
            "obs": obs_array,
            "state": state_array,
            "actions": actions_array,
            "old_logp": old_logp_array,
        }

        ppo_update(
            actor,
            critic,
            optimizer_actor,
            optimizer_critic,
            batch,
            advantages,
            returns,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            device=device,
        )

        buffer.clear()

        # Print status
        if len(episode_returns) > 0:
            avg_ret = sum(episode_returns) / len(episode_returns)
        else:
            avg_ret = 0.0
        elapsed = time.time() - start_time
        print(f"[T {timestep}] time {elapsed:.1f}s avg_return {avg_ret:.3f}")

    print("Training finished.")
    return actor, critic


# ---------------- Example runner ------------------
if __name__ == "__main__":
    # Quick test harness: replace ENV_FACTORY with a function returning your env
    try:
        import os
        import time

        ADJ = "basic_directed_network_1000.npy"
        run_name = f"MAPPO_shared_{ADJ}_{time.time()}"
        writer = SummaryWriter(f"runs/{run_name}")

        def env_factory():
            adjacency_matrix_path = os.path.join(
                os.path.dirname(__file__),
                "env_assets",
                ADJ,
            )
            adjacency_matrix = np.load(adjacency_matrix_path)
            return make_env(
                "sysadmin_network", **{"adjacency_matrix": adjacency_matrix}
            )

        actor, critic = train_shared_mappo(env_factory, writer)
    except RuntimeError as e:
        print("Please set ENV_FACTORY to return your environment instance. Error:", e)

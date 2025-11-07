import os
import random
import time
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Normal, Categorical
from torch.nn.utils import clip_grad_norm_
from gymnasium.spaces import MultiDiscrete

from torch.utils.tensorboard import SummaryWriter
from cognac.utils.make_env import make_env


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    debug_log: bool = False
    """if toggled, will log all power outputs and yaws step by step"""
    wandb_project_name: str = "COGNAC-benchmark-final"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = (
        "sysadmin_network"  # "binary_consensus"  # "grid_firefighting_graph" "sysadmin_network"
    )
    """the id of the environment"""
    adjacency_matrix_path = os.path.join(
        os.path.dirname(__file__), "env_assets", "basic_directed_network_10.npy"
    )
    """path to the adjacency matrix for the env - if relevant"""
    additional_env_params = {"max_steps": 100}
    """additionnal params for env instanciation"""
    total_timesteps: int = int(5e6)
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4  # 7e-4 #
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.9  # 0
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.04  # 0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    hidden_layer_nn: Union[bool, Union[bool, tuple[int, ...]]] = (32, 32)
    """number of neurons in hidden layer"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    buffer_size: int = 8
    """the replay memory buffer size"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    device: str = "cpu"


class EpisodicBuffer:
    def __init__(self, env, buffer_size=512):
        self.agents = env.possible_agents
        self.ep_max_length = env.max_steps
        self.obs_space_dim = [
            (
                np.prod(env.observation_space(agent).shape)
                if isinstance(env.observation_space(agent), MultiDiscrete)
                else 1
            )
            for agent in env.possible_agents
        ]
        self.act_space_dim = [
            env.action_space(agent).n  # TODO : Handle Box act for MCF
            for agent in env.possible_agents
        ]
        # env.reset()
        self.state_dim = np.prod(env.state().shape)
        self.buffer_size = buffer_size  # In terms of trajectories

        self.obs_buffers = {
            agent_id: np.zeros(
                (self.buffer_size, self.ep_max_length, self.obs_space_dim[agent_id])
            )
            for agent_id in self.agents
        }
        self.next_obs_buffers = {
            agent_id: np.zeros(
                (self.buffer_size, self.ep_max_length, self.obs_space_dim[agent_id])
            )
            for agent_id in self.agents
        }
        self.act_buffers = {
            agent_id: np.zeros((self.buffer_size, self.ep_max_length))
            for agent_id in self.agents
        }
        self.rewards_buffers = {
            agent_id: np.zeros((self.buffer_size, self.ep_max_length))
            for agent_id in self.agents
        }
        self.done_buffers = {
            agent_id: np.ones((self.buffer_size, self.ep_max_length))
            for agent_id in self.agents
        }
        self.state_buffer = np.zeros(
            (self.buffer_size, self.ep_max_length, self.state_dim)
        )

        self._counter_fill = 0
        self.full = False

        self.reset_ep_buffer()

    def store_ep_data(self):
        for agent in self.agents:
            self.obs_buffers[agent][self._counter_fill] = self.obs_ep_buffer[agent]
            self.next_obs_buffers[agent][self._counter_fill] = self.next_obs_ep_buffer[
                agent
            ]
            self.act_buffers[agent][self._counter_fill] = self.act_ep_buffer[agent]
            self.rewards_buffers[agent][self._counter_fill] = self.rew_ep_buffer[agent]
            self.done_buffers[agent][self._counter_fill] = self.dones_ep_buffer[agent]
        self.state_buffer[self._counter_fill] = self.states_ep_buffer

        self._counter_fill += 1
        self.reset_ep_buffer()
        if self._counter_fill >= self.buffer_size:
            self.full = True
            self._counter_fill = 0

    def reset_ep_buffer(self):
        self.obs_ep_buffer = {
            agent: np.zeros((self.ep_max_length, self.obs_space_dim[agent]))
            for agent in self.agents
        }
        self.next_obs_ep_buffer = {
            agent: np.zeros((self.ep_max_length, self.obs_space_dim[agent]))
            for agent in self.agents
        }

        self.act_ep_buffer = {
            agent: np.zeros(self.ep_max_length) for agent in self.agents
        }
        self.rew_ep_buffer = {
            agent: np.zeros(
                self.ep_max_length,
            )
            for agent in self.agents
        }
        self.dones_ep_buffer = {
            agent: np.ones(self.ep_max_length) for agent in self.agents
        }
        self.states_ep_buffer = np.zeros((self.ep_max_length, self.state_dim))

    def add_step(self, obs, next_obs, act, rew, dones, state, t):
        for agent in self.agents:
            self.obs_ep_buffer[agent][t] = obs[agent]
            self.next_obs_ep_buffer[agent][t] = next_obs[agent]
            self.act_ep_buffer[agent][t] = act[agent]
            self.rew_ep_buffer[agent][t] = rew[agent]
            self.dones_ep_buffer[agent][t] = dones[agent]
        self.states_ep_buffer[t] = state

    def sample(self, batch_size):
        if batch_size == self.buffer_size:
            return (
                self.obs_buffers,
                self.next_obs_buffers,
                self.act_buffers,
                self.rewards_buffers,
                self.done_buffers,
                self.state_buffer,
            )

        else:
            sampled_ids = np.random.choice(self.buffer_size, size=batch_size)
            obs, next_obs, act, rew, dones = {}, {}, {}, {}, {}
            for agent in self.agents:
                obs[agent] = self.obs_buffers[agent][sampled_ids]
                next_obs[agent] = self.next_obs_buffers[agent][sampled_ids]
                act[agent] = self.act_buffers[agent][sampled_ids]
                rew[agent] = self.rewards_buffers[agent][sampled_ids]
                dones[agent] = self.done_buffers[agent][sampled_ids]
            states = self.state_buffer[sampled_ids]
            return obs, next_obs, act, rew, dones, states


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SharedCritic(nn.Module):
    def __init__(self, state_dim, hidden_layers):
        super().__init__()
        self.state_dim = state_dim
        input_layers = [state_dim] + list([256, 256])
        self.critic = nn.Sequential(
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], input_layers[1:])
            ],
            layer_init(nn.Linear(input_layers[-1], 1), std=1.0),
        )

    def get_value(self, x):
        # x = (x - self.observation_low) / (self.observation_high - self.observation_low)
        return self.critic(x)


class Agent(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_layers):
        super().__init__()
        action_dim = action_dim

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.observation_dim = observation_dim

        input_layers = [self.observation_dim] + list(hidden_layers)
        self.actor = nn.Sequential(
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], input_layers[1:])
            ],
            layer_init(nn.Linear(input_layers[-1], action_dim), std=1.0),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, x, action=None, deterministic=False):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


if __name__ == "__main__":
    init_phase = time.time()
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cpu")
    lr_update_cnt = 0
    # environment and agents
    config = {}
    if isinstance(args.adjacency_matrix_path, str):
        config["adjacency_matrix"] = np.load(args.adjacency_matrix_path)

    if isinstance(args.additional_env_params, dict):
        config.update(args.additional_env_params)
    env = make_env(args.env_id, **config)
    env.reset()  # Reset to initialize state to get the shape.

    hidden_layer_nn = [64, 64]
    state_dim = np.prod(env.state().shape)
    agents = [
        Agent(
            int(np.prod(env.observation_space(j).shape)),
            env.action_space(j).n,
            hidden_layer_nn,
        ).to(device)
        for j in range(env.n_agents)
    ]

    shared_critic = SharedCritic(state_dim, hidden_layer_nn).to(device)
    actor_optimizers = [
        optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        for agent in agents
    ]
    critic_optimizer = optim.Adam(
        shared_critic.parameters(), lr=args.learning_rate / 10, eps=1e-5
    )

    # Instantiate Buffer
    buffer = EpisodicBuffer(env, buffer_size=args.buffer_size)

    def compute_gae(rewards, dones, values, next_values, gamma, gae_lambda):
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(rewards.shape[1])):
            nextnonterminal = 1.0 - dones[:, t]
            delta = (
                rewards[:, t]
                + gamma * next_values[:, t] * nextnonterminal
                - values[:, t]
            )
            advantages[:, t] = lastgaelam = (
                delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        return advantages, returns

    print(f"initialization {time.time() - init_phase}")
    global_step = 0
    filling_buffer = time.time()
    while global_step < args.total_timesteps:
        buffer.reset_ep_buffer()
        start_time = time.time()
        obs, _ = env.reset()
        episodic_return = 0
        for step in range(env.max_steps):
            with torch.no_grad():
                actions, logprobs = {}, {}
                for i, agent in enumerate(env.agents):
                    obs_tensor = torch.tensor(
                        obs[agent], dtype=torch.float32, device=device
                    )
                    action, logprob, _ = agents[i].get_action(obs_tensor)
                    actions[agent] = action.squeeze(0).cpu().numpy()
                    logprobs[agent] = logprob.cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = {
                agent: terminations[agent] or truncations[agent]
                for agent in env.possible_agents
            }
            state = (
                torch.tensor(env.state().flatten(), dtype=torch.float32).cpu().numpy()
            )

            buffer.add_step(obs, next_obs, actions, rewards, dones, state, step)
            global_step += 1
            obs = next_obs
            episodic_return += sum(rewards.values())

            if all(dones.values()):
                writer.add_scalar(
                    "global_measure/episodic_return",
                    episodic_return,
                    global_step,
                )
                episodic_return = 0
                break

        buffer.store_ep_data()
        if buffer.full:
            print(f"filling_buffer {time.time() - filling_buffer}")
            epoch_time = time.time()
            # ----------------------------------
            # Sample trajectories
            obs_batch, next_obs_batch, act_batch, rew_batch, done_batch, state_batch = (
                buffer.sample(args.buffer_size)
            )
            # TRAINING LOOP STARTS HERE
            # ==================================
            # Convert numpy batches to tensors
            device = shared_critic.critic[0][0].weight.device  # ensure consistency
            obs_b = {
                agent: torch.tensor(
                    obs_batch[agent], dtype=torch.float32, device=device
                ).reshape(-1, agents[agent].observation_dim)
                for agent in env.possible_agents
            }
            acts_b = {
                agent: torch.tensor(
                    act_batch[agent], dtype=torch.long, device=device
                ).flatten()
                for agent in env.possible_agents
            }
            rews_b = {
                agent: torch.tensor(
                    rew_batch[agent], dtype=torch.float32, device=device
                )
                for agent in env.possible_agents
            }
            dones_b = {
                agent: torch.tensor(
                    done_batch[agent], dtype=torch.float32, device=device
                )
                for agent in env.possible_agents
            }
            states = torch.tensor(
                state_batch, dtype=torch.float32, device=device
            ).reshape(-1, state_dim)

            # Compute values for all timesteps
            with torch.no_grad():
                values = shared_critic.get_value(states).reshape(
                    args.buffer_size, -1
                )  # (B, T)
                next_states = states[1:]  # .reshape(args.buffer_size, -1, state_dim)
                next_values = torch.cat(
                    [values[:, 1:], torch.zeros_like(values[:, :1])], dim=1
                )

            # Compute advantages and returns per agent
            advantages = {}
            returns = {}
            for agent in env.possible_agents:
                adv, ret = compute_gae(
                    rews_b[agent].reshape(args.buffer_size, -1),
                    dones_b[agent].reshape(args.buffer_size, -1),
                    values,
                    next_values,
                    args.gamma,
                    args.gae_lambda,
                )
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                advantages[agent] = adv.reshape(-1)
                returns[agent] = ret.reshape(-1)

            # Prepare old logprobs
            with torch.no_grad():
                old_logprobs = {}
                for i, agent in enumerate(env.possible_agents):
                    logits = agents[i](obs_b[agent])
                    dist = Categorical(logits=logits)
                    old_logprobs[agent] = dist.log_prob(acts_b[agent])

            # Flatten all experiences: already flattened via reshape above
            total_samples = args.buffer_size * env.max_steps
            minibatch_size = total_samples // args.num_minibatches

            # KL annealing LR scheduler (if desired)
            if args.anneal_lr:
                frac = 1.0 - (global_step / args.total_timesteps)
                lr_now = args.learning_rate * frac
                for opt in actor_optimizers + [critic_optimizer]:
                    for pg in opt.param_groups:
                        pg["lr"] = lr_now

            for _ in range(args.update_epochs):

                # generate shuffled indices
                idxs = torch.randperm(total_samples)
                for start in range(0, total_samples, minibatch_size):
                    mb_idx = idxs[start : start + minibatch_size]
                    # Critic update
                    mb_states = states[mb_idx]
                    mb_returns = (
                        torch.stack(
                            [returns[agent][mb_idx] for agent in env.possible_agents]
                        )
                        .mean(dim=0, keepdim=True)
                        .squeeze()
                    )
                    mb_values = shared_critic.get_value(mb_states).squeeze()
                    v_loss_unclipped = F.mse_loss(mb_values, mb_returns)
                    if args.clip_vloss:
                        v_clipped = values.reshape(-1)[mb_idx] + torch.clamp(
                            mb_values - values.reshape(-1)[mb_idx],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = F.mse_loss(v_clipped, mb_returns)
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                    else:
                        v_loss = 0.5 * v_loss_unclipped
                    critic_optimizer.zero_grad()
                    v_loss.backward()
                    clip_grad_norm_(shared_critic.parameters(), args.max_grad_norm)
                    critic_optimizer.step()

                    # Actor updates per agent
                    for i, agent in enumerate(env.possible_agents):
                        mb_obs = obs_b[agent][mb_idx]
                        mb_actions = acts_b[agent][mb_idx]
                        mb_adv = advantages[agent][mb_idx]
                        mb_oldlog = old_logprobs[agent][mb_idx]

                        logits = agents[i](mb_obs)
                        dist = Categorical(logits=logits)
                        mb_newlog = dist.log_prob(mb_actions)
                        ratio = torch.exp(mb_newlog - mb_oldlog)
                        surr1 = ratio * mb_adv
                        surr2 = (
                            torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            * mb_adv
                        )
                        p_loss = -torch.min(surr1, surr2).mean()
                        ent_loss = -args.ent_coef * dist.entropy().mean()

                        actor_optimizers[i].zero_grad()
                        (p_loss + ent_loss).backward()
                        clip_grad_norm_(agents[i].parameters(), args.max_grad_norm)
                        actor_optimizers[i].step()

            buffer = EpisodicBuffer(env, buffer_size=args.buffer_size)
            print(f"epoch {time.time() - epoch_time}")
            # Log metrics
            logging = time.time()
            writer.add_scalar("loss/value_loss", v_loss.item(), global_step)
            writer.add_scalar("loss/policy_loss", p_loss.item(), global_step)
            writer.add_scalar("loss/entropy_loss", ent_loss.item(), global_step)
            print(f"logging {time.time() - logging}")
            filling_buffer = time.time()

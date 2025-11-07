# The code in this file is directly adapted from https://github.com/ifpen/wfcrl-benchmark.

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gymnasium.spaces import Discrete, MultiDiscrete
from stable_baselines3.common.buffers import ReplayBuffer
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
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "COGNAC-benchmark-final"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    # Algorithm specific arguments
    env_id: str = "sysadmin_network"  # "binary_consensus"  # "grid_firefighting_graph"
    """the id of the environment"""
    adjacency_matrix_path = os.path.join(
        os.path.dirname(__file__), "env_assets", "basic_directed_network_100.npy"
    )
    """path to the adjacency matrix for the env - if relevant"""
    additional_env_params = None  # {"max_steps": 100}
    """additionnal params for env instanciation"""
    total_timesteps: int = 20_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.5
    """the target network update rate"""
    target_network_frequency: int = 20
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.005
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.1
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


class QMixer(nn.Module):
    def __init__(self, num_agents, state_space_dim, hidden_dim):
        super().__init__()
        """"
        Excerpt from QMIX paper
            "The mixing network consists of a single hidden layer of 32 units, utilising an ELU non-linearity. 
            
            The hypernetworks are then sized to produce weights of appropriate size. 
            The hypernetwork producing the final bias of the mixing network consists of a single hidden layer 
            of 32 units with a ReLU non-linearity
            
            Each hypernetwork takes the states as input and generates the weights of one layer of the mixing network. 
            Each hypernetwork consists of a single linear layer, followed by an absolute activation function, to
            ensure that the mixing network weights are non-negative.

            The output of the hypernetwork is then a vector, which is reshaped into a matrix of appropriate size. 
            The biases are produced in the same manner but are not restricted to being non-negative. 
            The final bias is produced by a 2 layer hypernetwork with a ReLU non-linearity."
        """
        self.input_dim = state_space_dim
        self.hidden_dim = hidden_dim[0]
        self.num_agents = num_agents

        # Hypernetworks
        self.hyper_network_w1 = layer_init(
            nn.Linear(self.input_dim, num_agents * self.hidden_dim),
            std=1.0,
        )
        self.hyper_network_b1 = layer_init(
            nn.Linear(self.input_dim, self.hidden_dim), std=1.0
        )

        self.hyper_network_w2 = layer_init(
            nn.Linear(self.input_dim, self.hidden_dim), std=1.0
        )
        self.hyper_network_b2 = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.hidden_dim), std=1.0),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )

    def forward(self, qvalues, state):
        """
        qvalues (batch_n, num_agent) B, M
        state   (batch_n, dim)
        """
        x = state
        w1 = torch.abs(self.hyper_network_w1(x))
        w1 = w1.view(-1, self.num_agents, self.hidden_dim)
        w2 = torch.abs(self.hyper_network_w2(x))
        # w2 = w2.view(-1, self.hidden_dim)
        b1 = self.hyper_network_b1(x)
        # b1 = b1.view(1, self.hidden_dim)
        b2 = self.hyper_network_b2(x)

        # QMixer network
        out = F.elu(qvalues[:, None, :] @ w1 + b1[:, None, :])
        out = out @ w2[:, :, None] + b2[:, :, None]
        return out

        # self.network[0].weight.data.copy_(w1.reshape(self.num_agents, self.hidden_dim))
        # self.network[0].bias.data.copy_(b1.flatten())

        # self.network[2].weight.data.copy_(w2.flatten())
        # self.network[2].bias.data.copy_(b2.flatten())


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims):
        super().__init__()

        self.action_dim = (
            sum(action_space.nvec)
            if isinstance(action_space, MultiDiscrete)
            else action_space.n
        )
        self.action_space = action_space
        self.observation_space = observation_space
        self.hidden_dim = hidden_dims[0]
        self.l1 = layer_init(
            nn.Linear(
                np.prod(observation_space.shape) + self.action_dim, hidden_dims[0]
            ),
            std=1.0,
        )
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        if isinstance(action_space, MultiDiscrete):
            self.output_layers = nn.ModuleList(
                [nn.Linear(hidden_dims[-1], act_n) for act_n in action_space.nvec]
            )
        elif isinstance(action_space, Discrete):
            self.output_layers = nn.Linear(hidden_dims[-1], action_space.n)
        else:
            raise Exception(
                "The action space should be either a Discrete or MultiDiscrete gym space."
            )
        self.reset_hidden_state()

    def reset_hidden_state(self, batch_size=1):
        # https://github.com/oxwhirl/pymarl/blob/master/src/modules/agents/rnn_agent.py
        self.hidden_state = self.l1.weight.new(batch_size, self.hidden_dim).zero_()

    def forward(self, x):
        # x should be observation + previous action (see Qmix paper)
        x = F.relu(self.l1(x)).view(-1, self.hidden_dim)
        h = self.rnn(x, self.hidden_state.view(-1, self.hidden_dim))
        q = self.output_layers(h)
        self.hidden_state = h
        return q

    def get_input(self, obs, last_act):
        vect_act = np.zeros(self.action_dim)
        vect_act[int(last_act)] = 1.0
        return np.concatenate([obs, vect_act])


# Epsilon scheduler
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    lr_update_cnt = 0
    # environment and agents
    config = {}
    if isinstance(args.adjacency_matrix_path, str):
        config["adjacency_matrix"] = np.load(args.adjacency_matrix_path)

    if isinstance(args.additional_env_params, dict):
        config.update(args.additional_env_params)
    env = make_env(args.env_id, **config)
    env.reset()  # Reset to initialize state to get the shape.

    obs_space = [
        (
            env.observation_space(agent)
            if isinstance(env.observation_space(agent), MultiDiscrete)
            else 1
        )
        for agent in env.possible_agents
    ]
    act_space = [
        (
            env.action_space(agent)
            if isinstance(env.action_space(agent), MultiDiscrete)
            else env.action_space(agent).n
        )
        for agent in env.possible_agents
    ]
    agents = [
        {
            "q_network": QNetwork(
                env.observation_space(j), env.action_space(j), [32]
            ).to(device),
            "q_target": QNetwork(
                env.observation_space(j), env.action_space(j), [32]
            ).to(device),
        }
        for j, agent in enumerate(env.possible_agents)
    ]
    optimizers = [
        optim.Adam(agent["q_network"].parameters(), lr=args.learning_rate)
        for agent in agents
    ]
    [
        agent["q_target"].load_state_dict(agent["q_network"].state_dict())
        for agent in agents
    ]

    state_dim = np.prod(env.state().shape)
    qmixer = QMixer(len(env.possible_agents), state_dim, [32]).to(device)
    target_qmixer = QMixer(len(env.possible_agents), state_dim, [32]).to(device)

    optimizers_qmix = optim.Adam(
        list(qmixer.parameters())
        + [p for ag in agents for p in ag["q_network"].parameters()],
        lr=args.learning_rate,
    )

    # Instantiate Buffer
    buffer = EpisodicBuffer(env, buffer_size=args.buffer_size)

    obs, _ = env.reset(seed=args.seed)
    joint_act = {agent_i: 0.0 for agent_i in env.possible_agents}
    last_joint_act = {agent_i: 0 for agent_i in env.possible_agents}
    state = env.state().flatten()
    episodic_return = 0
    filling_buffer = time.time()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        for i, agent in enumerate(agents):
            if random.random() < epsilon:
                act = env.action_space(i).sample()
            else:
                input_q = agent["q_network"].get_input(obs[i], last_joint_act[i])
                q_values = agent["q_network"](
                    torch.tensor(input_q, dtype=torch.float32)
                ).to(device)

                act = torch.argmax(q_values).cpu().numpy()
            joint_act[i] = act
        last_joint_act = joint_act
        next_obs, rewards, terminations, truncations, info = env.step(joint_act)
        episodic_return += sum(rewards.values())

        # Store next obs
        real_next_obs = next_obs

        # Check terminations & truncations states
        if any(terminations.values()) or any(truncations.values()):
            buffer.store_ep_data()
            next_obs, _ = env.reset(seed=args.seed)
            writer.add_scalar(
                "global_measure/episodic_return", episodic_return, global_step
            )

            episodic_return = 0
            last_joint_act = {agent_i: 0 for agent_i in env.possible_agents}
            for agent in agents:
                agent["q_network"].reset_hidden_state()

        # Store last transition in the buffer
        buffer.add_step(
            obs,
            real_next_obs,
            joint_act.copy(),
            rewards,
            terminations,
            state,
            env.timestep,
        )
        obs = next_obs
        # Reset Joint Act
        joint_act = {agent_i: 0.0 for agent_i in env.possible_agents}

        # TRAINING LOGIC
        if (
            buffer.full
            and (any(terminations.values()) or any(truncations.values()))
            # and lr_update_cnt % args.train_frequency == 0
        ):
            print(f"filling_buffer {time.time() - filling_buffer}")
            training = time.time()
            # 1. Sample a batch of full episodes
            obs_batch, next_obs_batch, act_batch, rew_batch, done_batch, state_batch = (
                buffer.sample(args.batch_size)
            )
            # obs_batch, act_batch, etc. are dicts of shape [agent][batch, T, ...]
            # state_batch: [batch, T, state_dim]

            # Convert to tensors
            # For simplicity we flatten (batch, T) → (batch*T)
            batch_sz, T = args.batch_size, buffer.ep_max_length
            device = next(qmixer.parameters()).device

            # Prepare per-agent tensors
            q_values = []
            target_q_values = []
            loop_agent = time.time()
            for i, agent_dict in enumerate(agents):
                # obs: (batch, T, obs_dim)
                obs = torch.tensor(
                    obs_batch[i], dtype=torch.float32, device=device
                )  # (B, T, obs_dim)
                obs_next = torch.tensor(
                    next_obs_batch[i], dtype=torch.float32, device=device
                )  # (B, T, obs_dim)
                actions = torch.tensor(
                    act_batch[i], dtype=torch.long, device=device
                )  # (B, T)
                dones = torch.tensor(
                    done_batch[i], dtype=torch.float32, device=device
                )  # (B, T)

                B, T, obs_dim = obs.shape
                n_actions = env.action_space(i).n

                # Prepare previous actions with zero-padding at t=0
                prev_actions = torch.zeros_like(actions)
                prev_actions[:, 1:] = actions[:, :-1]  # shift right
                prev_actions_oh = F.one_hot(
                    prev_actions, num_classes=n_actions
                ).float()  # (B, T, n_actions)

                # One-hot encode current actions (for q_input_next)
                actions_oh = F.one_hot(
                    actions, num_classes=n_actions
                ).float()  # (B, T, n_actions)

                # Concatenate obs and previous actions
                q_input = torch.cat([obs, prev_actions_oh], dim=-1)  # (B, T, obs+act)
                # q_input_next = torch.cat(
                #     [obs_next, actions_oh], dim=-1
                # )  # (B, T, obs+act)
                # Compute a_{t} → use it as a_{t-1} for obs_{t+1}
                next_prev_actions = torch.zeros_like(actions)
                next_prev_actions[:, 1:] = actions[:, :-1]
                next_prev_actions_oh = F.one_hot(
                    next_prev_actions, num_classes=n_actions
                ).float()
                q_input_next = torch.cat([obs_next, next_prev_actions_oh], dim=-1)

                # Initialize hidden states
                agent_dict["q_network"].reset_hidden_state(B)
                agent_dict["q_target"].reset_hidden_state(B)

                q_sa_seq = []
                q_next_seq = []

                for t in range(T):
                    # Current Q(s, a)
                    q_t = agent_dict["q_network"](q_input[:, t, :])  # (B, n_actions)
                    q_sa_t = q_t.gather(1, actions[:, t].unsqueeze(1)).squeeze(
                        1
                    )  # (B,)
                    q_sa_seq.append(q_sa_t)

                    # Target Q(s', max_a')
                    with torch.no_grad():
                        q_next_t = agent_dict["q_target"](
                            q_input_next[:, t, :]
                        )  # (B, n_actions)
                        q_max_t = q_next_t.max(dim=1)[0]  # (B,)
                        q_next_seq.append(q_max_t)

                q_sa_seq = torch.stack(q_sa_seq, dim=1)  # (B, T)
                q_next_seq = torch.stack(q_next_seq, dim=1)  # (B, T)

                q_values.append(q_sa_seq.view(-1))  # Flatten to (B*T,)
                target_q_values.append(q_next_seq.view(-1))  # Flatten to (B*T,)
            print(f"loop agent {time.time()-loop_agent}")
            # Stack per-agent: → (batch*T, num_agents)
            q_values = torch.stack(q_values, dim=1)
            target_q_values = torch.stack(target_q_values, dim=1)

            # Global state tensors: (batch, T, state_dim) → (batch*T, state_dim)
            states = torch.tensor(state_batch, dtype=torch.float32, device=device).view(
                -1, state_dim
            )
            # next state = we can shift states by 1 along T, but here just reuse state_batch
            next_states = states.clone()

            # Rewards: sum across agents or pick a global reward
            # Here we take the mean of per-agent rewards as the team reward
            rewards = torch.tensor(
                [list(rew_batch[i].flatten()) for i in range(len(agents))],
                dtype=torch.float32,
                device=device,
            ).sum(
                dim=0
            )  # → (batch*T,)

            # Done mask: any agent done → end of episode
            dones = torch.tensor(
                [list(done_batch[i].flatten()) for i in range(len(agents))],
                dtype=torch.float32,
                device=device,
            ).max(dim=0)[
                0
            ]  # → (batch*T,)

            # 2. Compute mixed Q and target mixed Q
            # online mix
            q_tot = qmixer(q_values, states).squeeze(-1)  # → (batch*T,)
            # target mix
            with torch.no_grad():
                q_tot_next = target_qmixer(target_q_values, next_states).squeeze(
                    -1
                )  # → (batch*T,)
                td_target = rewards + args.gamma * q_tot_next * (1.0 - dones)

            # 3. Compute loss and optimize
            loss = F.mse_loss(q_tot, td_target)

            optimizers_qmix.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qmixer.parameters(), 5.0)
            for ag in agents:
                torch.nn.utils.clip_grad_norm_(ag["q_network"].parameters(), 10)
            optimizers_qmix.step()

            writer.add_scalar("loss/td_loss", loss.item(), global_step)

            # Reset Env
            # TODO : improve variable naming
            obs, _ = env.reset(seed=args.seed)
            for agent in agents:
                agent["q_network"].reset_hidden_state()
                agent["q_target"].reset_hidden_state()

            # Soft update target networks
            if lr_update_cnt % args.target_network_frequency == 0:
                for ag in agents:
                    soft_update(ag["q_target"], ag["q_network"], args.tau)

                soft_update(target_qmixer, qmixer, args.tau)
            lr_update_cnt += 1
            print(f"training {time.time()-training}")
            filling_buffer = time.time()

# The code in this file is directly adapted from CleanRL's implementation of PPO.
# The original code can be found at : https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

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
    wandb_project_name: str = "COGNAC-benchmark"
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
    env_id: str = "grid_firefighting_graph"
    """the id of the environment"""
    adjacency_matrix_path = None  # os.path.join(os.path.dirname(__file__), "env_assets", "basic_undirected_network_10.npy")
    """path to the adjacency matrix for the env - if relevant"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_shape, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, act_dim),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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

    # environment and agents
    config = {}
    if isinstance(args.adjacency_matrix_path, str):
        config["adjacency_matrix"] = np.load(args.adjacency_matrix_path)

    env = make_env(args.env_id, **config)
    obs_space = [
        (
            env.observation_space(agent).shape[0]
            if isinstance(env.observation_space(agent), MultiDiscrete)
            else 1
        )
        for agent in env.possible_agents
    ]
    act_space = [
        (
            env.action_space(agent).shape[0]
            if isinstance(env.action_space(agent), MultiDiscrete)
            else env.action_space(agent).n
        )
        for agent in env.possible_agents
    ]
    # act_dim = env.single_action_space.n

    agents = [
        {
            "q_network": QNetwork(obs_space[j], act_space[j]).to(device),
            "q_target": QNetwork(obs_space[j], act_space[j]).to(device),
        }
        for j, agent in enumerate(env.possible_agents)
    ]
    optimizers = [
        optim.Adam(agent["q_network"].parameters(), lr=args.learning_rate)
        for agent in agents
    ]
    # q_network = QNetwork(envs).to(device)
    # optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    # target_network = QNetwork(envs).to(device)
    [
        agent["q_target"].load_state_dict(agent["q_network"].state_dict())
        for agent in agents
    ]

    rb = [
        ReplayBuffer(
            args.buffer_size,
            env.observation_space(i),
            env.action_space(i),
            device,
            handle_timeout_termination=False,
        )
        for i in range(len(agents))
    ]
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    episodic_return = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        actions = {}
        for i, agent in enumerate(agents):
            if random.random() < epsilon:
                act = env.action_space(i).sample()
            else:
                q_values = agent["q_network"](
                    torch.tensor(obs[env.possible_agents[i]], dtype=torch.float32).to(
                        device
                    )
                )
                act = torch.argmax(q_values).cpu().numpy()
            actions[env.possible_agents[i]] = act
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        episodic_return += sum(rewards.values())
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         if info and "episode" in info:
        #             # print(
        #             #     f"global_step={global_step}, episodic_return={info['episode']['r']}"
        #             # )
        #             writer.add_scalar(
        #                 "charts/episodic_return", info["episode"]["r"], global_step
        #             )
        #             writer.add_scalar(
        #                 "charts/episodic_length", info["episode"]["l"], global_step
        #             )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs
        if any(terminations.values()) or any(truncations.values()):
            next_obs, _ = env.reset(seed=args.seed)
            writer.add_scalar(
                "global_measure/episodic_return", episodic_return, global_step
            )
            episodic_return = 0
        [
            rb[j].add(
                obs[agent_name],
                real_next_obs[agent_name],
                actions[agent_name],
                rewards[agent_name],
                terminations[agent_name],
                infos[agent_name],
            )
            for j, agent_name in enumerate(env.possible_agents)
        ]

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        for i, (agent, optimizer) in enumerate(zip(agents, optimizers)):
            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                if global_step % args.train_frequency == 0:
                    data = rb[i].sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = agent["q_target"](
                            torch.tensor(data.next_observations, dtype=torch.float32)
                        ).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (
                            1 - data.dones.flatten()
                        )
                    old_val = (
                        agent["q_network"](
                            torch.tensor(data.observations, dtype=torch.float32)
                        )
                        .gather(1, data.actions)
                        .squeeze()
                    )
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar(
                            f"losses/agent_{i}/td_loss", loss, global_step
                        )
                        writer.add_scalar(
                            f"losses/agent_{i}/q_values",
                            old_val.mean().item(),
                            global_step,
                        )
                        # print("SPS:", int(global_step / (time.time() - start_time)))
                        # writer.add_scalar(
                        #     f"charts/agent_{i}/SPS",
                        #     int(global_step / (time.time() - start_time)),
                        #     global_step,
                        # )

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if global_step % args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(
                        agent["q_target"].parameters(), agent["q_network"].parameters()
                    ):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data
                            + (1.0 - args.tau) * target_network_param.data
                        )

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     torch.save(q_network.state_dict(), model_path)
    #     print(f"model saved to {model_path}")
    #     from cleanrl_utils.evals.dqn_eval import evaluate

    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=QNetwork,
    #         device=device,
    #         epsilon=0.05,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         writer.add_scalar("eval/episodic_return", episodic_return, idx)

    #     if args.upload_model:
    #         from cleanrl_utils.huggingface import push_to_hub

    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(
    #             args,
    #             episodic_returns,
    #             repo_id,
    #             "DQN",
    #             f"runs/{run_name}",
    #             f"videos/{run_name}-eval",
    #         )

    env.close()
    writer.close()

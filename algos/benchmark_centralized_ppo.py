# The code in this file is directly adapted from CleanRL's implementation of PPO.
# The original code can be found at : https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Discrete, MultiDiscrete

from cognac.utils.make_env import make_env


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
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
    env_id: str = "binary_consensus"
    """the id of the environment"""
    adjacency_matrix_path = os.path.join(
        os.path.dirname(__file__), "env_assets", "basic_undirected_network_10.npy"
    )
    """path to the adjacency matrix for the env - if relevant"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def to_bin(x, bits):
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])


class Agent(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(int(np.prod(obs_shape)), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(int(np.prod(obs_shape)), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    # def get_action_and_value(self, x, action=None):
    #     logits = self.actor(x)
    #     dist = Categorical(logits=logits)
    #     if action is None:
    #         action = dist.sample()
    #     return action, dist.log_prob(action), dist.entropy(), self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.sigmoid(logits)  # convert logits to probabilities for Bernoulli
        dist = Bernoulli(probs=probs)

        if action is None:
            action = dist.sample()

        # Important: Bernoulli's `log_prob` returns the log-prob per dimension
        # To get the total log_prob, sum over dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, self.critic(x)


if __name__ == "__main__":
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

    # compute derived sizes
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

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

    _, _ = env.reset()  # Ensure that the state is initiated
    agent = Agent(env.state().shape[0], env.n_agents).to(device)
    optimizers = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ==== STORAGE: one buffer per agent ====
    storage = {
        "obs": torch.zeros(args.num_steps, args.num_envs, env.state().shape[0]).to(
            device
        ),
        "actions": torch.zeros(args.num_steps, env.n_agents, args.num_envs).to(device),
        "logps": torch.zeros(args.num_steps, args.num_envs).to(device),
        "rews": torch.zeros(args.num_steps, args.num_envs).to(device),
        "done": torch.zeros(args.num_steps, args.num_envs).to(device),
        "vals": torch.zeros(args.num_steps, args.num_envs).to(device),
    }
    global_step = 0
    start_time = time.time()

    # reset returns a dict of agent→obs arrays of shape (num_envs, *obs_shape)
    next_obs_dict, _ = env.reset(seed=args.seed)
    # convert to per-agent tensors
    next_obs = torch.tensor(env.state(), dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32).to(device)

    total_rew = 0
    for it in range(1, args.num_iterations + 1):
        # optional LR annealing
        if args.anneal_lr:
            frac = 1 - (it - 1) / args.num_iterations
            for pg in optimizers.param_groups:
                pg["lr"] = frac * args.learning_rate

        # === ROLLOUT COLLECTION ===
        for t in range(args.num_steps):
            global_step += args.num_envs
            # record obs & dones
            storage["obs"][t] = next_obs
            storage["done"][t] = next_done

            # each agent acts
            actions_dict = {}
            with torch.no_grad():
                a, logp, _, val = agent.get_action_and_value(next_obs)
                storage["actions"][t] = a.unsqueeze(-1)
                storage["logps"][t] = logp
                storage["vals"][t] = val.view(-1)
                action = a.cpu().numpy()

                actions_dict = {
                    agent_name: act
                    for agent_name, act in zip(env.possible_agents, action)
                }

            # step environment
            nxt_obs_d, rews_d, terms_d, truncs_d, infos = env.step(actions_dict)
            total_rew += sum(rews_d.values())
            # unpack
            storage["rews"][t] = sum(rews_d.values())
            next_obs = torch.tensor(env.state(), dtype=torch.float32)
            next_done = any(terms_d.values()) or any(truncs_d.values())

            if next_done:
                init_obs, _ = env.reset()
                next_obs = torch.tensor(env.state(), dtype=torch.float32).to(device)
                next_done = False
                writer.add_scalar(
                    "global_measure/episodic_return", total_rew, global_step
                )
                total_rew = 0
            # (optional) log episode metrics from infos["final_info"]…

        # === COMPUTE ADVANTAGES & RETURNS & OPTIMIZE ===
        # for i, (agent_name, agent) in enumerate(zip(env.possible_agents, agents)):
        buf = storage
        # bootstrap last value
        with torch.no_grad():
            last_val = agent.get_value(next_obs).view(-1)
        # GAE
        adv = torch.zeros_like(buf["rews"])
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            nonterm = 1 - (
                buf["done"][t] if t == args.num_steps - 1 else buf["done"][t + 1]
            )
            nextval = last_val if t == args.num_steps - 1 else buf["vals"][t + 1]
            delta = buf["rews"][t] + args.gamma * nextval * nonterm - buf["vals"][t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + buf["vals"]

        # flatten
        b_obs = buf["obs"].reshape(-1, env.state().shape[0])
        b_acts = buf["actions"].reshape(-1, env.state().shape[0])
        b_logps = buf["logps"].reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = buf["vals"].reshape(-1)

        # optimize
        inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = inds[start : start + args.minibatch_size]
                na, nlp, ent, nv = agent.get_action_and_value(b_obs[mb], b_acts[mb])
                ratio = (nlp - b_logps[mb]).exp()

                # policy loss
                mb_adv = b_adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg1, pg2).mean()

                # value loss
                nv = nv.view(-1)
                if args.clip_vloss:
                    v_unclipped = (nv - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(
                        nv - b_val[mb], -args.clip_coef, args.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                    )
                else:
                    v_loss = 0.5 * ((nv - b_ret[mb]) ** 2).mean()

                loss = pg_loss - args.ent_coef * ent.mean() + args.vf_coef * v_loss

                optimizers.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizers.step()

            y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            writer.add_scalar(
                f"charts/learning_rate",
                optimizers.param_groups[0]["lr"],
                global_step,
            )
            writer.add_scalar(f"losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar(f"losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar(f"losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar(
                f"losses/explained_variance",
                explained_var,
                global_step,
            )

        # reset storage for next iteration
        for buf in storage.values():
            buf.zero_()

    env.close()
    writer.close()

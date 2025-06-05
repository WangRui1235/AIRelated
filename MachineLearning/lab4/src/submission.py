import math
import random
import json
import gymnasium as gym
import numpy as np
from safetensors.numpy import save_file, load_file
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from utils import StateT, ActionT, RLAlgorithm, Metrics


# 1
def valueIteration(
    succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]],
    discount: float,
    epsilon: float = 0.001,
):
    """
    Given transition probabilities and rewards, computes and returns V and
    the optimal policy pi for each state.

    Args:
        - succAndRewardProb: Dictionary mapping tuples of (state, action) to a list of (nextState, prob, reward) Tuples.
        - discount: The discount factor.
        - epsilon: The threshold at which to stop value iteration.

    Returns:
        Dictionary mapping each state to an action.
    """
    # Define a mapping from states to Set[Actions] so we can determine all the actions that can be taken from s.
    # You may find this useful in your approach.
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    def computeQ(V: Dict[StateT, float], state: StateT, action: ActionT) -> float:
        # Return Q(state, action) based on V(state)

        # BEGIN_YOUR_CODE
        # Return Q(state, action) based on V(state)
        q_value = 0.0
        # 遍历所有可能的下一个状态
        for nextState, prob, reward in succAndRewardProb[(state, action)]:
            # 根据Bellman方程累加: P(s'|s,a)[R(s,a,s') + γV(s')]
            q_value += prob * (reward + discount * V[nextState])
        return q_value
        # END_YOUR_CODE

    def computePolicy(V: Dict[StateT, float]) -> Dict[StateT, ActionT]:
        # Return the policy given V.
        # Remember the policy for a state is the action that gives the greatest Q-value.
        # IMPORTANT: if multiple actions give the same Q-value, choose the largest action number for the policy.
        # HINT: We only compute policies for states in stateActions.

        # 1-a
        # BEGIN_YOUR_CODE
        policy = {}
        for state in stateActions:
            # 使用 max() 找到 Q 值最大的动作，若有多个相同 Q 值则选择编号较大的动作
            best_action = max(stateActions[state], key=lambda action: (computeQ(V, state, action), action))
            policy[state] = best_action
    
        return policy
        # END_YOUR_CODE

    print("Running valueIteration...")
    V = defaultdict(float)  # This will return 0 for states not seen (handles terminal states)
    numIters = 0
    while True:
        newV = defaultdict(float)  # This will return 0 for states not seen (handles terminal states)
        # update V values using the computeQ function above.
        # repeat until the V values for all states do not change by more than epsilon.

        # 1-b
        # BEGIN_YOUR_CODE
        max_diff = 0
        # 遍历所有状态更新值函数
        for state in stateActions:
            # 找到该状态下所有动作中的最大Q值
            max_q = max(computeQ(V, state, action) for action in stateActions[state])
            newV[state] = max_q
            # 记录最大的值函数变化
            max_diff = max(max_diff, abs(newV[state] - V[state]))

        # 如果最大变化小于epsilon，则收敛
        if max_diff < epsilon:
            break
        # END_YOUR_CODE

        V = newV
        numIters += 1
    V_opt = V
    print(("valueIteration: %d iterations" % numIters))
    return computePolicy(V_opt)


# 2
class ModelBasedMonteCarlo(RLAlgorithm):
    def __init__(
        self,
        actions: List[ActionT],
        discount: float,
        calcValIterEvery: int = 10000,
        explorationProb: float = 0.2,
    ) -> None:
        self.actions = actions
        self.discount = discount
        self.calcValIterEvery = calcValIterEvery
        self.explorationProb = explorationProb
        self.numIters = 0

        # (state, action) -> {nextState -> ct} for all nextState
        self.tCounts = defaultdict(lambda: defaultdict(int))
        # (state, action) -> {nextState -> totalReward} for all nextState
        self.rTotal = defaultdict(lambda: defaultdict(float))

        self.pi = {}  # Optimal policy for each state. state -> action

    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        """
        This algorithm will produce an action given a state.

        Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
        Should return random action if the given state is not in self.pi.
        The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
        should always follow the policy if available.
        HINT: Use random.random() (not np.random()) to sample from the uniform distribution [0, 1]
        """
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:  # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e6:  # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        # 2-a
        # BEGIN_YOUR_CODE
        # 如果是测试模式且状态有策略，直接返回最优动作
        if not explore and state in self.pi:
            return self.pi[state]
        
        # 以explorationProb的概率选择随机动作
        if random.random() < explorationProb or state not in self.pi:
            return random.choice(self.actions)
        
        # 返回策略中的最优动作
        return self.pi[state]
        # END_YOUR_CODE

    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):
        """
        We will call this function with (s, a, r, s'), which is used to update tCounts and rTotal.
        For every self.calcValIterEvery steps, runs value iteration after estimating succAndRewardProb.

        Args:
            state: StateT, the current state
            action: ActionT, the action taken in the current state
            reward: int, the reward received after taking the action
            nextState: StateT, the next state
            terminal: bool, whether the episode is terminated
        """

        self.tCounts[(state, action)][nextState] += 1
        self.rTotal[(state, action)][nextState] += reward

        if self.numIters % self.calcValIterEvery == 0:
            # Estimate succAndRewardProb based on self.tCounts and self.rTotal.
            # Hint 1: prob(s, a, s') = (counts of transition (s,a) -> s') / (total transtions from (s,a))
            # Hint 2: Reward(s, a, s') = (total reward of (s,a) -> s') / (counts of transition (s,a) -> s')
            # Then run valueIteration and update self.pi.
            succAndRewardProb = defaultdict(list)  # (state, action) -> [(nextState, transitionProb, expectedreward)]

            # 2-b
            # BEGIN_YOUR_CODE
            succAndRewardProb = {
                (s, a): [
                    (ns, 
                    count/sum(nexts.values()),  # 转移概率
                    self.rTotal[(s, a)][ns]/count  # 期望奖励
                    )
                    for ns, count in nexts.items()
                ]
                for (s, a), nexts in self.tCounts.items()
            }

            # 更新策略
            self.pi = valueIteration(succAndRewardProb, self.discount)
            # END_YOUR_CODE

    def save_pretrained(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "actions": self.actions,
            "discount": self.discount,
            "calcValIterEvery": self.calcValIterEvery,
            "explorationProb": self.explorationProb,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        mcvi_weights = {str(k): np.array(v) for k, v in self.pi.items()}
        save_file(mcvi_weights, path / "mcvi.safetensors")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        pi = load_file(path / "mcvi.safetensors")
        mcvi = cls(**config)
        mcvi.pi = {eval(k): int(v.item()) for k, v in pi.items()}
        return mcvi


# 3
class TabularQLearning(RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float, explorationProb: float = 0.2, initialQ: float = 0):
        """
        Args:
            - actions: the list of valid actions
            - discount: a number between 0 and 1, which determines the discount factor
            - explorationProb: the epsilon value indicating how frequently the policy returns a random action
            - intialQ: the value for intializing Q values.
        """
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.initialQ = initialQ
        self.Q = defaultdict(lambda: initialQ)  # Dict[Tuple[Tuple[float, float], int], float]
        self.numIters = 0

    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        """
        This algorithm will produce an action given a state.
        Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
        The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
        should always choose the maximum Q-value action.
        """
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:  # explore
            explorationProb = 1.0
        elif self.numIters > 1e5:  # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        # HINT 1: You can access Q-value with self.Q[state, action]
        # HINT 2: Use random.random() to sample from the uniform distribution [0, 1]

        # 3-a
        # BEGIN_YOUR_CODE
        # ε-greedy策略实现
        # 先判断 Q 值是否为空，然后根据 explore 的条件来选择行动
        if not any(self.Q[(state, action)] for action in self.actions):
            return random.choice(self.actions)

        if explore and random.random() < explorationProb:
            return random.choice(self.actions)

        return max(self.actions, key=lambda action: self.Q[(state, action)])
        # END_YOUR_CODE

        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.1

    def incorporateFeedback(
        self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool
    ) -> None:
        """
        We will call this function with (s, a, r, s'), which you should use to update |Q|.
        Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
        You should update the Q values using self.getStepSize()
        HINT 1: The target V for the current state is a combination of the immediate reward
        and the discounted future value.
        HINT 2: V for terminal states is 0
        """

        # 3-b
        # BEGIN_YOUR_CODE
        # 获取学习率
        # 获取学习率 α
        step_size = self.getStepSize()

        # 计算TD目标值
        if terminal:  
            target = reward  # 终止状态只有即时奖励
        else:
            # 非终止状态：r + γ * max_a' Q(s',a')
            max_next_q = max(self.Q[(nextState, a)] for a in self.actions)
            target = reward + self.discount * max_next_q

        # TD更新：Q(s,a) ← Q(s,a) + α[target - Q(s,a)]
        self.Q[(state, action)] += step_size * (target - self.Q[(state, action)])
        # END_YOUR_CODE

    def save_pretrained(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "actions": self.actions,
            "discount": self.discount,
            "explorationProb": self.explorationProb,
            "initialQ": self.initialQ,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        Q_table = {f"{key[0][0]}, {key[0][1]}, {key[1]}": np.array(value) for key, value in self.Q.items()}
        save_file(Q_table, path / "tabular.safetensors")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        rl = cls(**config)
        loaded_data = load_file(path / "tabular.safetensors")
        Q_table = {((eval(key)[0], eval(key)[1]), eval(key)[2]): value.item() for key, value in loaded_data.items()}
        rl.Q = Q_table
        return rl


# If you don't want to try Policy Gradient, the above is enough for the submission.


# 4 [Optional] Policy Gradient
class Policy(nn.Module, PyTorchModelHubMixin):
    """
    Your Policy Network. It can actually work well for CartPole-v1 environment.
    But you can change the architecture if you want.

    Args:
        - state_dim: The dimension of the state space.
        - action_dim: The dimension of the action space.
        - h_size: The size of the hidden layer.

    Methods:
        - forward: Forward pass of the network.
        - getAction: Get the action from the policy network.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2, h_size: int = 24):
        super(Policy, self).__init__()

        self.state_projection = nn.Linear(state_dim, h_size)
        # self.hidden_layers = nn.Sequential(
        #     nn.Linear(h_size, h_size),
        #     nn.ReLU(),
        #     nn.Linear(h_size, h_size),
        #     nn.ReLU()
        # )
        self.action_head = nn.Linear(h_size, action_dim)

    def forward(self, x):
        x = F.relu(self.state_projection(x))
        # x = self.hidden_layers(x)
        x = self.action_head(x)
        x = F.softmax(x, dim=1)
        return x

    def getAction(self, state: torch.Tensor):
        """
        Get the action from the policy network.

        Args:
            - state: The input state tensor.

        Returns:
            - action: The action sampled from the policy network.
            - log_prob: The log probability of the action for policy gradient.
        """
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(
    policy: Policy,
    batch_size: int,
    lr: float,
    num_updates: int,
    max_t: int,
    gamma: float,
    checkpoint_path: Union[str, Path],
    window_size: int = 50,
    save_every: int = 100,
    metrics: Optional[Metrics] = None,
):
    """
    Reinforce Algorithm for Policy Gradient.

    Args:
        - policy: The policy network.
        - batch_size: The size of the batch.
        - lr: The learning rate.
        - num_updates: The number of updates.
        - max_t: The maximum time step.
        - gamma: The discount factor.
        - checkpoint_path: The path to save the checkpoint.
        - window_size: The size of the moving average window.
        - save_every: The frequency of saving the model.
        - metrics: The metrics object.

    Returns:
        - policy: The trained policy network.
    """
    num_params = sum(p.numel() for p in policy.parameters())
    num_params_k = num_params / 10**3
    print(f"Parameters : {num_params_k:.3f}K Total.")
    assert num_params_k < 100, "Your model is too large!"

    # Online RL Algorithm contains many CPU-GPU transfers, so it is better to use CPU only in this case.
    # But you can use GPU if you want, and make sure all of the tensors are in the same device.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    # policy.to(device)

    R_deque = deque(maxlen=window_size)  # Store the whole episode rewards.

    env = gym.make("CartPole-v1")

    status = metrics.get_status() if metrics is not None else "Start training!"
    with tqdm(total=num_updates, desc=status, leave=False) as pbar:
        for upd_step in range(1, num_updates + 1):
            opt.zero_grad()
            for i_episode in range(1, batch_size + 1):
                saved_log_probs = []
                rewards = []
                state, _ = env.reset()

                # 4-a
                # Sample an episode using Gymnasium environment and Policy Network.
                # Use policy.getAction(state) to get the action and log probability.
                # Use env.step(action) to get the next_state, reward, terminated, truncated and info.
                # You can look up the MountainCarMDP.transition() function as a reference.

                # BEGIN_YOUR_CODE
                for t in range(max_t):  # 最多运行max_t步
                    # 获取动作和对应的log概率
                    action, log_prob = policy.getAction(state)
                    saved_log_probs.append(log_prob)
                    
                    # 执行动作
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    rewards.append(reward)
                    
                    # 更新状态
                    state = next_state
                    
                    # 如果episode结束则退出
                    if terminated or truncated:
                        break
                # END_YOUR_CODE

                traj_r = sum(rewards)
                R_deque.append(traj_r)

                returns = deque(maxlen=max_t)  # G(t) = r(t) + gamma * G(t+1)
                n_steps = len(rewards)

                # 4-b
                # Compute the discounted return for each time step.

                # BEGIN_YOUR_CODE
                # 从后向前计算折扣回报
                R = 0  # 用于累积回报
                for t in reversed(range(n_steps)):
                    R = rewards[t] + gamma * R  # 计算当前时刻的折扣回报
                    returns.appendleft(R)  # 将回报添加到队列前端
                # END_YOUR_CODE

                # 4-c
                # Maybe standardization of the returns is employed to make training more stable
                # eps = np.finfo(np.float32).eps.item()
                # eps is the smallest representable float, which is
                # added to the standard deviation of the returns to avoid numerical instabilities.

                # BEGIN_YOUR_CODE
                # 标准化returns
                returns = torch.tensor(returns)
                eps = np.finfo(np.float32).eps.item()

                if len(returns) > 1:  # 只有当returns长度大于1时才进行标准化
                    returns = (returns - returns.mean()) / (returns.std() + eps)

                # 计算每个时间步的loss并累加
                policy_loss = []
                for log_prob, R in zip(saved_log_probs, returns):
                    policy_loss.append(-log_prob * R)  # 负号是因为我们要最大化期望奖励

                # 将所有时间步的loss加起来并除以batch_size进行归一化
                loss = torch.cat(policy_loss).sum() / batch_size
                # END_YOUR_CODE

                if metrics is not None:
                    # For a single episode, we only record the loss and reward.
                    metrics.commit(
                        loss=loss,
                        reward=traj_r,
                    )
                loss.backward()
            opt.step()

            if metrics is not None:
                metrics.commit(update_step_time=True, step=upd_step, lr=lr)
                status = metrics.push()

                if metrics.step % save_every == 0:
                    policy.save_pretrained(checkpoint_path / f"checkpoint_{metrics.step}")
            else:
                status = f"Step: {upd_step}, Average Reward: {np.mean(R_deque):.2f}"

                if upd_step % save_every == 0:
                    policy.save_pretrained(checkpoint_path / f"checkpoint_{upd_step}")

            # Update Pbar
            pbar.update()
            pbar.set_description(status)

    # Final Checkpoint
    policy.save_pretrained(checkpoint_path / "final")
    print(f"Final Model Saved at {checkpoint_path / 'final'}")

    return policy

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import copy
import random
import torch.nn.functional as F
import logging
import sys
from gym.wrappers.atari_preprocessing import AtariPreprocessing  #
from gym.wrappers.frame_stack import FrameStack  # 使用这个游戏
import gym
import matplotlib.pyplot as plt

TAU = 1e-3  # for soft update of target parameters
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.995  # discount factor
UPDATE_EVERY = 20  # how often to update the network
LearnOneTime = 33
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')
env =FrameStack(AtariPreprocessing(gym.make('PongNoFrameskip-v4')),#
                 num_stack=4)


class Buffer:
    def __init__(self,capacity):
        self.buffer = pd.DataFrame(index=range(capacity),
                                   columns=['state', 'action', 'reward', 'next_state', 'done'])
        self.i = 0
        self.capacity = capacity
        self.capacity_now = 0

    def remember(self, state, action, reward, next_state, done):
        self.buffer.loc[self.i] = (state, action, reward, next_state, done)
        self.i = (self.i + 1) % self.capacity
        self.capacity_now = min(self.capacity,self.capacity_now + 1)

    def example(self,size):
        sample = np.random.choice(self.capacity_now, size=size)
        out = (np.stack(self.buffer.loc[sample, field]) for field in self.buffer.columns)
        return out

    # def sample(self, size):
    #     indices = np.random.choice(self.capacity_now, size=size)
    #     out = (np.stack(self.buffer.loc[indices, field]) for field in self.buffer.columns)  # 将数组合并
    #     return out


class Agent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.actionspace = env.action_space
        self.gamma = 0.99
        self.epsilon = 1.  # exploration

        self.memory = Buffer(capacity=100000)

        # self.atom_count = 51
        # self.atom_min = -10.
        # self.atom_max = 10.
        # self.atom_difference = (self.atom_max - self.atom_min) \
        #                        / (self.atom_count - 1)
        # self.atom_tensor = torch.linspace(self.atom_min, self.atom_max,
        #                                   self.atom_count)  # 输出等距的值（tensor）

        self.evaluate_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(inplace=True),
            nn.Linear(512, self.action_n ))
        self.target_net = copy.deepcopy(self.evaluate_net)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=0.0001)
        self.mode = 'train'
        self.trajectory = []
        self.t_step = 0
        self.eps = 0.95
    def reset(self, mode=None):
        self.mode = mode
        if mode == 'train':
            self.trajectory = []

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.remember(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.capacity_now > BATCH_SIZE:
                experiences = self.memory.example(LearnOneTime)
                #experienc1 = self.memory.sample(LearnOneTime)
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given5 state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #state = torch.tensor(state)..unsqueeze(0)
        state = torch.as_tensor(state,dtype=torch.float).unsqueeze(0)
        self.evaluate_net.eval()
        with torch.no_grad():
            action_values = self.evaluate_net(state)
        self.evaluate_net.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            out = random.choice(np.arange(self.action_n))
            #print('\r',out)
            return out

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        state_tensor = torch.as_tensor(states, dtype=torch.float)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        done_tensor = torch.as_tensor(dones, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_state_tensor).detach()
        # Compute Q targets for current states   此处should be rewrite
        Q_targets = reward_tensor.unsqueeze(1).repeat(1,6) + (gamma * Q_targets_next * (1 - done_tensor.unsqueeze(1).repeat(1,6)))

        # Get expected Q values from local model
        Q_expected = self.evaluate_net(state_tensor)
            #.gather(1, actions_tensor)  #####

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.evaluate_net, self.target_net, TAU)
        self.eps = max(self.eps - 0.02 , 0.1)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


agent = Agent(env)


# 训练&测试
def play_episode(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0., False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.act(observation)
        preobserve = observation
        #print(action)

        if render:
            env.render()
        if done:
            break
        observation, reward, done, _ = env.step(action)
        agent.step(preobserve, action, reward, observation, done)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            break
    #agent.close()
    return episode_reward, elapsed_steps


logging.info('==== train ====')
episode_rewards = []

episode0 = 0
while 1:
    episode0 += 1
    episode_reward, elapsed_steps = play_episode(env, agent, mode='train', render=1)
    episode_rewards.append(episode_reward)
    logging.debug('train episode %d: reward = %.2f, steps = %d',
                  episode0, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-5:]) > 16.:
        break
plt.plot(episode_rewards)

logging.info('==== test ====')
episode_rewards = []

for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent, render=1)
    episode_rewards.append(episode_reward)
    logging.debug('test episode %d: reward = %.2f, steps = %d',
                  episode, episode_reward, elapsed_steps)
logging.info('average episode reward = %.2f ± %.2f',
             np.mean(episode_rewards), np.std(episode_rewards))
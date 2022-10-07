# import gym
# env = gym.make('CartPole-v1')
# print('观测空间 = {}'.format(env.observation_space))
# print('动作空间 = {}'.format(env.action_space))
# print('动作数 = {}'.format(env.action_space.n))
# print('初始状态 = {}'.format(env.state))
# init_state = env.reset()
# print('初始状态 = {}'.format(init_state))
# print('初始状态 = {}'.format(env.state))
# env.render(mode='human')

# import gym
# env = gym.make("CartPole-v1")
# observation, info = env.reset(seed=42)
#
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(observation)
#
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()
# \
# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
# for t in range(100):
#     env.render()
#     print(observation)
#     action = env.action_space.sample()
#     observation, reward, done, info,_ = env.step(action)
#     if done:
#         print("Episode finished after {} timesteps".format(t+1))
#         break
# env.close()


import gym
import NN
# def policy():


env = gym.make("LunarLander-v2")
agent = NN.Agent(8, 4, 2016)
observation = env.reset()
episode_reward = 0
#epoch = 1
for epoch in range(10000):
    while 1:

        action = agent.act(observation)  # User-defined policy function
        preob = observation
        observation, reward, terminated, truncated = env.step(action)
        episode_reward = reward + episode_reward
        #print(episode_reward)
        agent.step(preob, action, reward, observation, terminated)
        # print(observation)
        # if reward > 0: print('输出：%.2f' % reward)
        if terminated or truncated:
            observation, info = env.reset()
            print("epoch:",epoch,' reward:',episode_reward)
            #epoch += 1
            episode_reward = 0
            break

env.close()

# import gym
# env = gym.make("LunarLander-v2", render_mode="human")
# env.action_space.seed(42)
# #
# # observation, info = env.reset(seed=42)
# #
# # for _ in range(1000):
# #     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
# #
# #     if terminated or truncated:
# #         observation, info = env.reset()
# #
# # env.close()
#
# from gym.utils.env_checker import check_env
#
# check_env(env)


# import gym
# env = gym.make('CartPole-v1',render_mode='human')
# for i_episode in range(2):
#     observation = env.reset()
#     #初始话环境
#     for t in range(1000):
#         env.render()
#         #提供环境
#         action = env.action_space.sample()
#         #在可行的动作空间中随机选择一个
#         observation, reward, done, info, _ = env.step(action)
#         #顺着这个动作进入下一个状态
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#
# env.close()

# import gym
# import time
# env = gym.make('CartPole-v1',render_mode='human')   #创造环境
# observation = env.reset()       #初始化环境，observation为环境状态
# count = 0
# for t in range(100):
#     action = env.action_space.sample()  #随机采样动作
#     observation, reward, done, info, _ = env.step(action)  #与环境交互，获得下一步的时刻
#     # if done:
#     #     break
#     env.render()         #绘制场景
#     count+=1
#     time.sleep(0.2)      #每次等待0.2s
# print(count)             #打印该次尝试的步数
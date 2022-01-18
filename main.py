from config import *
from ddpg import DDPG
import gym
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # initialize the environment
    env = gym.make(ENV_NAME)
    env = env.unwrapped

    # set random seed for reproducible
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # define state dimension, action dimension and action bound
    state_dim = env.observation_space.shape[0]
    print(env.action_space)
    action_dim = env.action_space.shape[0]

    agent = DDPG(state_dim, action_dim)

    # ------------------------------------ train ------------------------------------ #
    reward_buffer = []  # record the reward of every episode
    t0 = time.time()  # record current time
    var = VAR
    for i in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0  # record the reward of current episode
        t1 = time.time()
        for j in range(MAX_EP_STEPS):

            # choose an action and add exploration noise
            action = agent.choose_action(state)
            action = np.clip(np.random.normal(action, var), -2, 2)

            # interact with environment
            next_state, reward, done, info = env.step(action)

            # store transition
            agent.observe(state, action, reward / 10, next_state)

            # start learning if the memory is full
            if agent.is_memory_full():
                var *= .9995
                agent.learn()

            state = next_state
            ep_reward += reward

            episode_time = time.time() - t1
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i,
                      ' Reward: %.2f' % ep_reward,
                      ' Explore: %.2f' % var,
                      ' Time: %.2f' % float(episode_time))
                reward_buffer.append(ep_reward)
                break
            if SHOW_PLT:
                plt.show()

        if SHOW_PLT and reward_buffer:
            plt.ion()
            plt.cla()
            plt.title('DDPG Reward')
            plt.plot(np.array(range(len(reward_buffer))), reward_buffer)  # plot the episode vt
            plt.xlabel('episode steps')
            plt.ylabel('normalized reward')
            plt.show()
            plt.pause(0.1)

    train_time = time.time() - t0
    plt.ioff()
    plt.show()
    print('\n Running time: ', train_time)
    # ------------------------------------------------------------------------------- #

    # ------------------------------------ test ------------------------------------- #
    while True:
        state = env.reset()
        for i in range(MAX_EP_STEPS):
            env.render()
            state, reward, done, info = env.step(agent.choose_action(state))
    # ------------------------------------------------------------------------------- #

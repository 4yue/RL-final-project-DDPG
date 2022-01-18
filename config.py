""" hyper parameters """

ENV_NAME = 'Pendulum-v1'  # environment name
RANDOM_SEED = 1  # random seed

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
BATCH_SIZE = 32  # update batch size

MEMORY_CAPACITY = 10000  # size of replay buffer

MAX_EPISODES = 200  # total number of episodes for training
MAX_EP_STEPS = 200  # total number of steps for each episode

VAR = 3  # control exploration

SHOW_PLT = True

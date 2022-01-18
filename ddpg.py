import torch
from torch.optim import Adam
from model import *
from config import *
from memory import *
from util import *
import os

criterion = nn.MSELoss()


class DDPG(object):
    def __init__(self, state_dim, action_dim):
        """
        initialize some parameters
        :param state_dim: dimension of state
        :param action_dim: dimension of action
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # build actor network
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.actor_opt = Adam(self.actor.parameters(), lr=LR_A)  # optimizer

        # build critic network
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        self.critic_opt = Adam(self.critic.parameters(), lr=LR_C)  # optimizer

        # update parameters, make sure that the parameters of the 2 networks is same
        copy_para(self.actor_target, self.actor)
        copy_para(self.critic_target, self.critic)

        # replay buffer
        self.memory = Memory(MEMORY_CAPACITY, 2 * self.state_dim + self.action_dim + 1)

        # hyper parameters
        self.batch_size = BATCH_SIZE
        self.discount_factor = GAMMA
        self.tau = TAU

    def learn(self):
        """
        update parameters
        """
        bt = self.memory.sample(self.batch_size)
        batch_state = bt[:, :self.state_dim]
        batch_action = bt[:, self.state_dim:self.state_dim + self.action_dim]
        batch_reward = bt[:, -self.state_dim - 1:-self.state_dim]
        batch_next_state = bt[:, -self.state_dim:]

        # Critic:
        next_action = self.actor_target(to_tensor(batch_next_state))
        next_q_values = self.critic_target([
            to_tensor(batch_next_state),
            next_action,
        ])
        target_q = to_tensor(batch_reward) + self.discount_factor * next_q_values

        self.critic.zero_grad()

        q = self.critic([to_tensor(batch_state), to_tensor(batch_action)])

        value_loss = criterion(q, target_q)
        value_loss.backward()
        self.critic_opt.step()

        # Actor:
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(batch_state),
            self.actor(to_tensor(batch_state))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_opt.step()

        # soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def observe(self, state, action, reward, next_state):
        """
        store (state, action, reward, next_state) into memory and update the state
        :param state: current state
        :param action: current action
        :param reward: current reward
        :param next_state: next moment state
        """
        self.memory.store(state, action, reward, next_state)

    def choose_action(self, state):
        """
        :return: an action chosen from action network
        """
        action = self.actor(to_tensor(np.array([state])))
        action = to_numpy(action).squeeze(0)
        return action

    def save_model(self):
        """
        save model to disk
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(self.actor.state_dict(), 'saved_model/actor.pkl')
        torch.save(self.critic.state_dict(), 'saved_model/critic.pkl')

    def load_model(self):
        """
        load model from disk
        """
        self.actor.load_state_dict(torch.load('saved_model/actor.pkl'))
        self.critic.load_state_dict(torch.load('saved_model/critic.pkl'))

    def is_memory_full(self):
        """
        :return: true if memory is full else false
        """
        if self.memory.index < self.memory.size:
            return False
        return True

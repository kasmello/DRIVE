import torch
import torch.nn as nn
import pygame
import os
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box, Dict
from tqdm import tqdm


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        super().__init__()
        # create network layers
        layers = nn.ModuleList()

        # input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # output layer (preferences/logits/unnormalised log-probabilities)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # combine layers into feed-forward network
        self.net = nn.Sequential(*layers)

        # select optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        # return output of policy network
        return self.net(x)
    
    def update(self, states, actions, returns):
        # update network weights for a given transition or trajectory
        self.optimizer.zero_grad()
        logits = self.net(states)
        dist = torch.distributions.Categorical(logits=logits)
        loss = torch.mean(-dist.log_prob(actions)*returns)
        loss.backward()
        self.optimizer.step()

# Value network for approximating value function
class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, learning_rate):
        super().__init__()
        # create network layers
        layers = nn.ModuleList()

        # input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # output layer (there is only one unit representing state value)
        layers.append(nn.Linear(hidden_sizes[-1], 1))

        # combine layers into feed-forward network
        self.net = nn.Sequential(*layers)

        # select loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        # return output of value network for the input x
        return self.net(x)

    def update(self, inputs, targets):
        # update network weights for given input(s) and target(s)
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

# Actor-Critic algorithm with one-step TD target
class ActorCritic():
    def __init__(self, env, gamma, hidden_sizes=(32, 32), lr_policy=0.001, lr_value=0.001):
        # check if the state space has correct type
        # continuous = isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 1
        # assert continuous, f'Observation space must be continuous with shape (n,), current space is {env.observation_space} n'
        self.state_dims = len(env.deconstruct_array(env.reset()))


        # check if the action space has correct type
        # assert isinstance(env.action_space, Discrete), 'Action space must be discrete'
        self.num_actions = 9

        # create policy network
        self.policynet = PolicyNetwork(self.state_dims, hidden_sizes, self.num_actions, lr_policy)

        # create value network
        self.valuenet = ValueNetwork(self.state_dims, hidden_sizes, lr_value)

        self.env = env
        self.gamma = gamma

    def policy(self, state, stochastic=True):
        # convert state to torch format
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        # calculate action probabilities
        logits = self.policynet(state).detach()
        dist = torch.distributions.Categorical(logits=logits)
        if stochastic:
            # sample action using action probabilities
            return dist.sample().item()
        else:
            # select action with the highest probability
            # note: we ignore breaking ties randomly (low chance of happening)
            return dist.probs.argmax().item()

    def update(self, state, action, reward, next_state, terminated):
        # calculate TD target for value network update
        if terminated:
            target = reward
        else:
            target = reward + self.gamma*self.valuenet(next_state).detach()

        # convert target to torch format
        target = torch.tensor([target], dtype=torch.float)

        # calculate TD error for policy network update (equal to the action advantage)
        delta = target - self.valuenet(state).detach()

        # update networks
        action = torch.tensor(action)
        self.policynet.update(state, action, delta)
        self.valuenet.update(state, target)

    def train(self, max_episodes, criterion_episodes, folder_path):
        # train the agent for a number of episodes
        num_steps = 0
        episode_rewards = []
        for episode in range(max_episodes):
            state = self.env.reset()

            # convert state to torch format
            state = self.env.deconstruct_array(state)
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            truncated = False
            episode_rewards.append(0)
            while not (terminated or truncated):
                # select action by following policy
                action = self.policy(state)

                # send the action to the environment
                next_state, reward, terminated, _ = self.env.step(action)
                frame = self.env.render()
                frame_name = f'{episode}.png'
                pygame.image.save(frame, os.path.join(folder_path, frame_name))
                episode_rewards[-1] += reward

                # convert next state to torch format
                next_state = torch.tensor(self.env.deconstruct_array(next_state), dtype=torch.float)

                # update policy and value networks
                self.update(state, action, reward, next_state, terminated)

                state = next_state
                num_steps += 1

            print(f'\rEpisode {episode+1} done: steps = {num_steps}, '
                  f'rewards = {episode_rewards[episode]}     ', end='')

            if episode >= criterion_episodes-1:
                print(f'\nStopping criterion satisfied after {episode} episodes')
                break

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards)+1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def save(self, path):
        # save network weights to a file
        torch.save({'policy': self.policynet.state_dict(),
                    'value': self.valuenet.state_dict()}, path)

    def load(self, path):
        # load network weights from a file
        networks = torch.load(path)
        self.policynet.load_state_dict(networks['policy'])
        self.valuenet.load_state_dict(networks['value'])
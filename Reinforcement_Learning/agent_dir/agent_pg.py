import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import os, json
#import matplotlib.pyplot as plt

from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')
        # discounted reward
        self.gamma = 0.99 
        
        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        self.draw_freq = 50
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.log_probs = []
    
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []
        self.log_probs = []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        # TODO:
        # discount your saved reward
        tmp = 0
        decay_reward = []
        loss = []
        for r in self.rewards[::-1]:
            tmp = r + self.gamma * tmp
            decay_reward.append(tmp)
        decay_reward = torch.tensor(decay_reward[::-1])
        decay_reward = (decay_reward - decay_reward.mean()) / decay_reward.std()

        # TODO:
        # compute loss
        
        for prob, r in zip(self.log_probs, decay_reward):
            loss.append(-prob * r)
        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()

        loss.backward()
        self.optimizer.step()

#    def save_curve(self, x_values, y_values, title):
#
#        tmp = {title:
#                {
#                    'x': x_values,
#                    'y': y_values
#                }
#            }
#
#        if os.path.isfile('./policy.json'):
#            with open('policy.json', 'r') as f:
#                file = json.load(f)
#            file.update(tmp)
#            with open('policy.json', 'w') as f:
#                json.dump(file, f)
#        else:
#            with open('policy.json', 'w') as f:
#                json.dump(tmp, f)

    def train(self):
        avg_reward = None # moving average of reward
        x = []
        y = []
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                
                self.saved_actions.append(action)
                self.rewards.append(reward)

            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
            if epoch % self.draw_freq == 0:
                x.append(epoch)
                y.append(avg_reward)

            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break
#        self.save_curve(x, y, 'pg_baseline')

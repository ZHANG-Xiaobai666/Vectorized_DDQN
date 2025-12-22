
import torch.nn.functional as F
import torch
import torch.nn as nn
from algorithms.actor import Actor
import numpy as np
import copy

import math

class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.device = args.dvc

        self.num_channel = args.num_channel
        self.num_agent = args.num_agent
        self.max_branch_num = args.max_num_branch

        self.actor_input_dim = int(args.max_num_branch * (args.num_channel + 1)) + args.max_num_branch
        self.hid_dim_lstm = args.hid_dim_lstm
        self.hid_dim_v = args.hid_dim_v
        self.actor = Actor(self.actor_input_dim, self.hid_dim_lstm, self.hid_dim_v, self.num_channel, self.max_branch_num)
        self.actor_lr = args.actor_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)

        self.actor_eval = Actor(self.actor_input_dim, self.hid_dim_lstm, self.hid_dim_v, self.num_channel, self.max_branch_num)
        self.actor_eval.load_state_dict(self.actor_eval.state_dict())

        self.max_epsilon = args.epsilon      # exploration probability
        self.epsilon = self.max_epsilon
        self.gamma = args.gamma



    def select_action(self, obs, branch_num, rnn_state, deterministic=False):
        #obs = check(obs).to(**self.tpdv)
        obs = np.concatenate([obs[0], obs[1][:, None]], axis=-1)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).view(1, 1, self.actor_input_dim)

        q_value, rnn_state = self.actor(obs, rnn_state)  # q_value  Kmax x (channel + 1)
        q_eff = q_value[:branch_num]
        q_masked = q_eff.clone()
        action_selected = np.zeros(branch_num)
        q_selected = [[] for _ in range(branch_num)]
        branch_ava = [i for i in range(branch_num)]
        if deterministic:
            for idx in range(min(branch_num, self.num_channel)):
                idx = torch.argmax(q_masked)
                branch = idx // q_masked.size(-1)
                action_idx = idx % q_masked.size(-1)
                if action_idx != 0 and self.num_channel < self.num_agent:
                    if np.random.random() < 1 - max(self.num_channel, self.max_branch_num) / self.num_agent:
                        action_idx = torch.zeros(1)
                action_selected[branch] = action_idx
                q_selected[branch] = q_eff[branch][0][0][action_idx]
                q_masked[branch, 0, 0, :] = -float("inf")
                branch_ava.remove(branch)
        else:
            if np.random.rand() < self.epsilon:
                num_select_channels = min(branch_num, self.num_channel)
                indices = np.random.choice(branch_num, size=num_select_channels, replace=False)
                for branch in indices:
                    action_idx = np.random.randint(0,self.num_channel+1)
                    action_selected[branch] = action_idx
                    q_selected[branch] = q_eff[branch][0][0][action_idx]
                    branch_ava.remove(branch)
            else:
                for idx in range(min(branch_num, self.num_channel)):
                    idx = torch.argmax(q_masked)
                    branch = idx // q_masked.size(-1)
                    action_idx = idx % q_masked.size(-1)
                    if action_idx != 0 and self.num_channel < self.num_agent:
                        if np.random.random() < 1 - max(self.num_channel, self.max_branch_num)/self.num_agent:
                            action_idx = torch.zeros(1)
                    action_selected[branch] = action_idx
                    q_selected[branch] = q_eff[branch][0][0][action_idx]
                    q_masked[branch, 0, 0, :] = -float("inf")
                    branch_ava.remove(branch)

        return q_selected, action_selected, rnn_state

    def eval_action(self, obs, rnn_state, branch_num, action_to_eval):   #
        q_selected_eval = [[] for _ in range(branch_num)]
        with torch.no_grad():
            obs = np.concatenate([obs[0], obs[1][:, None]], axis=-1)
            obs = torch.tensor(np.array(obs), dtype=torch.float32).view(1, 1, self.actor_input_dim)
            q_value, _ = self.actor(obs, rnn_state)
            q_eval_value, _ = self.actor_eval(obs, rnn_state)
            for branch in action_to_eval:
                action_idx = q_value[branch, 0, 0, :].argmax()
                q_selected_eval[branch] = q_eval_value[branch, 0, 0, action_idx]
        return q_selected_eval

    def random_select_action(self,  obs, branch_num, rnn_state):
        obs = np.concatenate([obs[0], obs[1][:, None]], axis=-1)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).view(1, 1, self.actor_input_dim)

        q_value, rnn_state = self.actor(obs, rnn_state)  # q_value  Kmax x (channel + 1)
        q_eff = q_value[:branch_num]
        action_selected = np.zeros(branch_num)
        q_selected = [[] for _ in range(branch_num)]
        branch_ava = [i for i in range(branch_num)]
        num_select_channels = min(branch_num, self.num_channel)
        indices = np.random.choice(branch_num, size=num_select_channels, replace=False)
        for branch in indices:
            action_idx = np.random.randint(0, self.num_channel + 1)
            action_selected[branch] = action_idx
            q_selected[branch] = q_eff[branch][0][0][action_idx]
            branch_ava.remove(branch)
        return action_selected, rnn_state
    def train_mini_batch(self, batchQ):
        actual_Qs = torch.stack([torch.stack(a) for (a, t) in batchQ])
        traget_Qs = torch.stack([torch.stack(t) for (a, t) in batchQ])
        q_loss = F.mse_loss(actual_Qs, traget_Qs).mean()
        #print(q_loss)

        self.actor_optimizer.zero_grad()
        q_loss.backward()
        self.actor_optimizer.step()


    def update_eval(self):
        self.actor_eval.load_state_dict(self.actor.state_dict())


    def save(self, save_dir):
        torch.save(self.actor.state_dict(), save_dir)

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(load_dir, weights_only=True,  map_location=self.device))
        self.actor.eval()

    def exploration_decay(self, episode, episodes):
        self.epsilon = self.max_epsilon - (self.max_epsilon * (episode / float(episodes)))
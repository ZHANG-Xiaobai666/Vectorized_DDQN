
import time
import numpy as np
import torch
from algorithms.agent import Agent
import os
import collections
import random
import copy





class EnvRunner():
    def __init__(self, args, env):
        self.episodes = args.episode_num
        self.episode_length = args.episode_length
        self.train_every_step = args.train_every_step
        self.update_eval_every_step = args.update_eval_every_step
        self.save_eval_every_step = args.save_eval_every_step
        self.env = env

        self.num_agent = args.num_agent                     # total agents in the network
        self.num_channel = args.num_channel
        self.max_num_branch = args.max_num_branch
        self.gamma = args.gamma
        #self.num_active_num_agent = args.num_agent          # active agents in the network

        self.save_dir = args.save_dir


        self.agents = []
        for _ in range(self.num_agent):
            agent = Agent(args)
            self.agents.append(agent)

        self.buffers = [[] for _ in range(self.num_agent)]


    def run(self):
        start = time.time()

        for episode in range(self.episodes):

            self.env.reset()
            obs_next = self.env.get_obs()
            obs = copy.deepcopy(obs_next)
            rnn_states = [None for _ in range(self.num_agent)]
            for step in range(self.episode_length):

                q_values, actions, rnn_states = self.collect(obs, rnn_states)             # collect actions and corresponding probs
                obs_next, rewards = self.env.step(actions, step)

                self.push(q_values, obs_next, rnn_states, rewards, step)
                obs = copy.copy(obs_next)

                if (step + 1) % self.train_every_step == 0:
                    self.train()
                    self.buffers = [[] for _ in range(self.num_agent)]
                    rnn_states = [None for _ in range(self.num_agent)]
                    self.update_par(step + 1, self.episode_length)

                    if (step + 1) % self.update_eval_every_step == 0:
                        self.eval_update()
                        if (step + 1) % self.save_eval_every_step == 0:
                            self.save_model()

                    print(f"Iteration: {step + 1} / {self.episode_length}")
                    print(f"Throughput {self.env.get_short_term_throughput(step + 1)}")

            #.update_par(step, self.episode_length)



        end = time.time()
    def collect(self, obs, rnn_states):
        actions = []
        next_rnn_states = []
        q_values = []
        curr_branch = min(self.num_agent, self.max_num_branch)
        for agent in range(self.num_agent):
            q_value, action, rnn_state = self.agents[agent].select_action(obs[agent], curr_branch, rnn_states[agent])
            q_values.append(q_value)
            actions.append(action)
            next_rnn_states.append(rnn_state)
        return q_values, actions, next_rnn_states

    def push(self, q_values, obs_next, rnn_states, rewards, step):
        curr_branch = min(self.num_agent, self.max_num_branch)
        for agent in range(self.num_agent):
            action_idx = [i for i, x in enumerate(q_values[agent]) if x != []]
            q_eval = self.agents[agent].eval_action(obs_next[agent], rnn_states[agent], curr_branch, action_idx)
            target_q = [rewards[agent][i] + self.gamma*x if x !=[] else x for i, x in enumerate(q_eval)]
            q_values[agent] = [x for x in q_values[agent] if x]
            target_q = [x for x in target_q if x]
            self.buffers[agent].append((q_values[agent], target_q))

    def train(self):
        for agent in range(self.num_agent):
            self.agents[agent].train_mini_batch(self.buffers[agent])

    def eval_update(self):
        for agent in range(self.num_agent):
            self.agents[agent].update_eval()

    def update_par(self, episode, episodes):
        for agent in range(self.num_agent):
            self.agents[agent].exploration_decay(episode, episodes)


    def save_model(self):
        for agent in range(self.num_agent):
            self.agents[agent].save(os.path.join(self.save_dir, "agent" + str(agent) + ".pt"))

    def load_model(self):
        for agent in range(self.num_agent):
            self.agents[agent].load(os.path.join(self.save_dir, "agent" + str(agent) + ".pt"))


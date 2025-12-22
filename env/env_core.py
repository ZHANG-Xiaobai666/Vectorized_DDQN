import random
import numpy as np

class EnvCore:

    def __init__(self, args, arr_pro):
        self.episode_length = args.episode_length       # Number of time slot per train/execute
        self.num_branch = args.max_num_branch
        self.num_agent = args.num_agent                 # Number of nodes may be dynamic from 0 - max_num_branch
        self.arr_pro = arr_pro                          # aggregate arrival probability
        self.num_channel = args.num_channel                            # Number of channels

        self.queue_length = np.zeros(self.num_agent)   # backlogged packets

        self.num_successful_packets = np.zeros(self.num_agent) # successful transmitted packets
        self.throughput = 0                 # network throughput sum(num_successful_packets)/episode_length)

        self.obs_dim = args.obs_dim

        self.obs = [(np.zeros((self.obs_dim[0][0], self.obs_dim[0][1])), np.zeros(self.obs_dim[1])) for _ in range(
            self.num_agent)]    # ((channel+1, branch), branch) -> (action, reward)

        self.time_slot = 0

    def reset(self):

        self.queue_length = np.zeros(self.num_agent)    #  backlogged packets
        self.num_successful_packets = np.zeros(self.num_agent)
        self.throughput = 0

        self.obs = [(np.zeros((self.obs_dim[0][0], self.obs_dim[0][1])), np.zeros(self.obs_dim[1])) for _ in range(
            self.num_agent)]    # ((channel+1, branch), branch) -> (action, reward)

        self.time_slot = 0
    def step(self, actions, step):  # time: from step to step + min(num_agent, max_num_branch) -1

        actions = np.array(actions)
        deci_time = min(self.num_agent, self.num_branch)
        rewards = [np.zeros(deci_time) for _ in range(self.num_agent)]  # 1 success -1 collision 0 no transmission
        self.obs = [(np.zeros((self.obs_dim[0][0], self.obs_dim[0][1])), np.zeros(self.obs_dim[1])) for _ in range(
            self.num_agent)]    # ((channel+1, branch), branch) -> (action, reward)

        for time in range(deci_time):

            self.time_slot += 1

            """ packet generation """
            random_numbers = np.array([random.random() for _ in range(self.num_agent)])
            index = np.where(random_numbers < self.arr_pro)[0]
            for idx in index:
                self.queue_length[idx] += 1

            actions_time = actions[:, time]

            """ get feedback/reward from the channel and update obs based actions and rewards """
            for ch_idx in range(self.num_channel):
                x = np.where(actions_time == ch_idx + 1)[0]
                if len(x) == 1:                          # successful if only one node transmits
                    rewards[x[0]][time] = 1
                    self.num_successful_packets[x[0]] += 1
                    self.queue_length[x[0]] -= 1
                    self.obs[x[0]][0][time][ch_idx] = 1   # update action in obs
                    self.obs[x[0]][1][time] = 1           # update reward in obs
                elif len(x) > 1:                   # collision
                    for idx in range(len(x)):
                        rewards[x[idx]][time] = -1
                        self.obs[x[idx]][0][time][ch_idx] = 1   # update action in obs
                        self.obs[x[idx]][1][time] = -1          # update reward in obs

            x = np.where(actions_time == 0)[0]         # idle
            for idx in range(len(x)):
                rewards[x[idx]][time] = 0
                self.obs[x[idx]][0][time][0] = 1  # update action in obs
                self.obs[x[idx]][1][time] = 0     # update reward in obs

            """update number of active nodes"""

        if step == self.episode_length-1:
            self.throughput = sum(self.num_successful_packets)/self.time_slot/self.num_channel

        return self.obs, rewards

    def get_throughput(self):
        return self.throughput


    def get_sum_success(self):
        return self.num_successful_packets

    def get_short_term_throughput_for_each(self):
        return self.num_successful_packets/self.time_slot

    def get_short_term_throughput(self):
        return sum(self.num_successful_packets)/self.time_slot

    def get_obs(self):
        return self.obs

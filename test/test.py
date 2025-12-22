import numpy as np
import time
import argparse
import os

from env.env_core import EnvCore
from run.env_test_runner import EnvRunner
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--episode_num', type=int, default=int(1000), help='num of episodes')
parser.add_argument('--episode_length', type=int, default=int(200), help='number of time slots per episode')
parser.add_argument('--train_every_step', type=int, default=int(1), help='num of steps to train DQN')
parser.add_argument('--update_eval_every_step', type=int, default=int(2), help='num of steps to update eva DQN')
parser.add_argument('--save_eval_every_step', type=int, default=int(4), help='num of steps to save DQN')


parser.add_argument('--epsilon', type=float, default=0.1, help='exploration probability')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
parser.add_argument('--minimal_batch_size', type=int, default=int(4e3), help='batch size for training')



parser.add_argument('--learning_interval', type=int, default=int(100), help='learning interval for each training')
parser.add_argument('--update_interval', type=int, default=int(200), help='update interval for the evaluate network')




parser.add_argument('--hid_dim_lstm', type=int, default=int(300), help='Hidden net width of lstm layer')
parser.add_argument('--hid_dim_v', type=int, default=int(50), help='Hidden net width of state value layer')
parser.add_argument('--critic_net_width', type=int, default=int(128), help='Hidden net width of critic network')

parser.add_argument('--actor_lr', type=float, default=0.01, help='Learning rate')



parser.add_argument('--max_num_branch', type=int, default=5, help='kmax in the paper')





args = parser.parse_args()

def make_env(args, arr_pro):
    return EnvCore(args, arr_pro)

def main():

    num_agent = int(5)
    num_channel = int(2)
    arr_pro = 1

    args.num_agent = num_agent
    args.num_channel = num_channel

    args.action_dim = args.max_num_branch*(args.num_channel+1)
    args.obs_dim = ((args.max_num_branch, args.num_channel+1), args.max_num_branch)  # (action, rewards)

    root_path = os.path.dirname(os.getcwd())
    args.save_dir = os.path.join(root_path, "results")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # args.save_dir = os.path.join(args.save_dir, args.reward_type)
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, "N" + str(args.num_agent) + "K" + str(args.num_channel))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    env = make_env(args, arr_pro)
    runner = EnvRunner(args, env)
    runner.run()





if __name__ == "__main__":
    main()

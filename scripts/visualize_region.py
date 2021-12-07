#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy, load_feasibiltiy
from safe_rl.utils.logx import EpochLogger

def collect_obs(env, bound=1.0):
    env.reset()
    config_dict = env.world_config_dict
    x = np.linspace(-bound, bound, 10)
    y = np.linspace(-bound, bound, 10)
    X, Y = np.meshgrid(x, y)
    obs = []
    for i in range(10):
        for j in range(10):
            print(i, j)
            config_dict['robot_xy'] = np.array([X[i, j], Y[i, j]])
            env.world_config_dict = config_dict
            env.world.rebuild(config_dict)
            obs.append(env.obs())
            env.render()
    a = 1
    return obs

def visualize_region(get_feasibility):
    pass

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    from custom_env_utils import register_custom_env
    register_custom_env()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='/home/mahaitong/PycharmProjects/safety-starter-agents/data/2021-11-20_cpo_Safexp-CustomGoal2-v0/2021-11-20_18-41-41-cpo_Safexp-CustomGoal2-v0_s3')
    parser.add_argument('--len', '-l', type=int, default=None)
    parser.add_argument('--episodes', '-n', type=int, default=5)
    parser.add_argument('--norender', '-nr', action='store_true', default=False)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    env, get_feasibility_indicator, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))

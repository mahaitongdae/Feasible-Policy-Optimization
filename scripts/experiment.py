#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from custom_env_utils import register_custom_env


def main(env_id, algo, seed, exp_name, cpu):
    register_custom_env()

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian','ppo_dual_ascent', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    # task = task.capitalize()
    # robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    # assert task.lower() in task_list, "Invalid task"
    # assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + env_id
    num_steps = 1.2e6
    steps_per_epoch = 8000
    cost_lim = 10
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    # exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = env_id

    algo(env_fn=lambda: gym.make(env_name),
         ac_kwargs=dict(
             hidden_sizes=(64, 64),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--robot', type=str, default='Car')
    # parser.add_argument('--task', type=str, default='Goal2')
    parser.add_argument('--env_id', type=str, default='Safexp-CustomPush2-v0')
    parser.add_argument('--algo', type=str, default='ppo_dual_ascent')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='for exp')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.env_id, args.algo, args.seed, exp_name, args.cpu)

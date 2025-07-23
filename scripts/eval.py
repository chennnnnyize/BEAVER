import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

import argparse
import numpy as np
import torch
import pickle
from a2c_ppo_acktr.model import Policy
from environments.building.env_building import BuildingEnv_2d, BuildingEnv_3d, BuildingEnv_9d
from environments.building.env_dynamic_building import BuildingEnv_DR_2d, BuildingEnv_DR_3d, BuildingEnv_DR_9d
from environments.building.utils_building import ParameterGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='results/Building-dr-3d/cmorl-ipo/0/final', help='Path to directory containing saved models')
    parser.add_argument('--save-dir', type=str, default='results/Building-dr-3d/cmorl-ipo')
    parser.add_argument('--eval-num', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--building', type=str, default='OfficeLarge')
    parser.add_argument('--weather', type=str, default='Warm_Marine')
    parser.add_argument('--location', type=str, default='ElPaso')
    parser.add_argument('--env-name', type=str, default='building_dr_3d')
    parser.add_argument('--obj-num', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-seed', type=int, default=0)
    parser.add_argument('--layernorm', action='store_true',default=False, help='if use layernorm')
    return parser.parse_args()


def load_policy(args, env, policy_path):
    policy = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={'recurrent': False, 'layernorm' : args.layernorm},
        obj_num=args.obj_num)
    policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
    policy.eval()
    return policy

def evaluation(args, task, eval_env, actor_critic, env_params):
    objs = np.zeros(args.obj_num)
    policy = actor_critic
    ob_rms = env_params['ob_rms']
    with torch.no_grad():
        for eval_id in range(args.eval_num):
            eval_env.seed = args.seed + eval_id
            #ob = eval_env.reset(seed = args.seed + eval_id)[0]
            ob = eval_env.reset_with_task(task = task, seed = args.seed + eval_id)[0]
            done = False
            gamma = 1.0
            while not done:
                ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
                _, action, _, _ = policy.act(torch.Tensor(ob).unsqueeze(0), None, None, deterministic=True)
                action = action.cpu().numpy()
                action = np.squeeze(action)
                ob, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                info['obj'] = reward
                objs += gamma * info['obj']
                gamma *= 1.0
    eval_env.close()
    objs /= args.eval_num
    return objs

def generate_task_list_from_bounds(n_tasks=5, seed=0):
    np.random.seed(seed)
    bounds_mean = [
        (0.774, 6.299),     # intwall
        (0.386, 3.145),     # floor
        (0.269, 2.191),     # outwall
        (0.160, 1.304),     # roof
        (0.386, 3.145),     # ceiling
        (0.386, 3.145),     # groundfloor
        (1.950, 3.622),     # window
    ]
    task_list = []
    for _ in range(n_tasks):
        task = [np.random.uniform(low, high) for (low, high) in bounds_mean]
        task_list.append(task)
    return task_list
    
def main_():
    args = parse_args()

    i = 0
    all_results = []
    while True:
        model_path = os.path.join(args.model_dir, f'EP_policy_{i}.pt')
        env_param_path = os.path.join(args.model_dir, f'EP_env_params_{i}.pkl')
        if not os.path.exists(model_path) or not os.path.exists(env_param_path):
            break

        print(f"Evaluating policy {i}...")

        # Load saved environment parameters
        with open(env_param_path, 'rb') as fp:
            env_params = pickle.load(fp)

        # Initialize environment
        if args.env_name == 'building_dr_2d':
            params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
            env = BuildingEnv_DR_2d(params)
        elif args.env_name == 'building_dr_3d':
            params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
            env = BuildingEnv_DR_3d(params)
        elif args.env_name == 'building_dr_9d':
            params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
            env = BuildingEnv_DR_9d(params)
        elif args.env_name == 'building_2d':
            params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
            env = BuildingEnv_2d(params)
        elif args.env_name == 'building_3d':
            params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
            env = BuildingEnv_3d(params)
        elif args.env_name == 'building_9d':
            params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
            env = BuildingEnv_9d(params)
        else:
            raise NotImplementedError(f"Unknown env: {args.env_name}")

        # Load policy
        model = load_policy(args, env, model_path)

        # Evaluate
        avg_reward = evaluation(args, env, model, env_params)
        print(f"Policy {i} avg_reward: {avg_reward}")
        all_results.append((i, avg_reward))
        i += 1
        
def main():
    args = parse_args()
    
    task_list = generate_task_list_from_bounds()
    print(task_list)
    for t, task in enumerate(task_list):
        all_results = []
        i = 0
        print(task)
        while True:
            model_path = os.path.join(args.model_dir, f'EP_policy_{i}.pt')
            env_param_path = os.path.join(args.model_dir, f'EP_env_params_{i}.pkl')
            if not os.path.exists(model_path) or not os.path.exists(env_param_path):
                break

            print(f"Evaluating policy {i}...")

            # Load saved environment parameters
            with open(env_param_path, 'rb') as fp:
                env_params = pickle.load(fp)

            # Initialize environment
            if args.env_name == 'building_dr_2d':
                params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
                env = BuildingEnv_DR_2d(params)
            elif args.env_name == 'building_dr_3d':
                params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
                env = BuildingEnv_DR_3d(params)
            elif args.env_name == 'building_dr_9d':
                params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
                env = BuildingEnv_DR_9d(params)
            elif args.env_name == 'building_2d':
                params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
                env = BuildingEnv_2d(params)
            elif args.env_name == 'building_3d':
                params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
                env = BuildingEnv_3d(params)
            elif args.env_name == 'building_9d':
                params = ParameterGenerator(Building=args.building, Weather=args.weather, Location=args.location)
                env = BuildingEnv_9d(params)
            else:
                raise NotImplementedError(f"Unknown env: {args.env_name}")

            # Load policy
            model = load_policy(args, env, model_path)

            # Evaluate
            avg_reward = evaluation(args, task, env, model, env_params)
            print(f"Policy {i} avg_reward: {avg_reward}")
            all_results.append(avg_reward)
            i += 1

        all_results = np.array(all_results)

        save_path = os.path.join(args.save_dir, 'final')
        os.makedirs(save_path, exist_ok=True)
        
        print(all_results, all_results.shape)

        with open(os.path.join(save_path, 'eval_obj_task'+str(t)+'_seed_'+str(args.run_seed)+'_.txt'), 'w') as fp:
            for row in all_results:
                fp.write(('{:5f}' + (len(row) - 1) * ',{:5f}' + '\n').format(*row))




if __name__ == '__main__':
    main()

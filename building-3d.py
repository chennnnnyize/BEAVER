import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cmorl-ipo', default=False, action='store_true')
parser.add_argument('--cmorl-cpo', default=False, action='store_true')
parser.add_argument('--random', default=False, action='store_true')
parser.add_argument('--num-seeds', type=int, default=6)
parser.add_argument('--save-dir', type=str, default='./results/Building-3d')
parser.add_argument('--ref-point', type=float, nargs='+', default=[0., 0., 0.])
args = parser.parse_args()

random.seed(2000)
save_dir = args.save_dir

test_cmorl_ipo = args.cmorl_ipo
test_cmorl_cpo = args.cmorl_cpo

for i in range(args.num_seeds):
    seed = random.randint(0, 1000000)
    print(f"Running seed: {seed}")
    
    if test_cmorl_ipo:
        cmd = f'python morl/run.py '\
            f'--env-name building_3d --obj-num 3 '\
            f'--seed {seed} '\
            f'--num-time-steps 2000000 '\
            f'--num-init-steps 1500000 '\
            f'--ref-point 0 0 0 '\
            f'--min-weight 0.0 '\
            f'--max-weight 1.0 '\
            f'--delta-weight 0.5 '\
            f'--eval-delta-weight 0.1 '\
            f'--eval-num 10 '\
            f'--eval-gamma 1.0 '\
            f'--num-select 6 '\
            f'--update-method cmorl-ipo '\
            f'--obj-rms '\
            f'--ob-rms '\
            f'--raw '\
            f'--save-dir {save_dir}/cmorl-ipo/{i}/'
        
        print("Running CMORL-IPO")
        ret_code = os.system(cmd)
        if ret_code != 0:
            print("CMORL-IPO execution failed")
            break

    if test_cmorl_cpo:
        cmd = f'python morl/run.py '\
            f'--env-name building_3d --obj-num 3 '\
            f'--seed {seed} '\
            f'--num-time-steps 2000000 '\
            f'--num-init-steps 1500000 '\
            f'--ref-point 0 0 0 '\
            f'--min-weight 0.0 '\
            f'--max-weight 1.0 '\
            f'--delta-weight 0.25 '\
            f'--eval-num 1 '\
            f'--num-select 9 '\
            f'--update-method cmorl-cpo '\
            f'--obj-rms '\
            f'--ob-rms '\
            f'--raw '\
            f'--save-dir {save_dir}/cmorl-cpo/{i}/'
        
        print("Running CMORL-CPO")
        ret_code = os.system(cmd)
        if ret_code != 0:
            print("CMORL-CPO execution failed")
            break

print("All executions completed")
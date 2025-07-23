# BEAVER
Code repo for Building Environments with Assessable Variation for Evaluating Multi-Objective Reinforcement Learning

Authors: Ruohong Liu, Jack Umenberger, and Yize Chen

## Introduction

While individual success is observed in simulated or controlled environments, the scalability of RL approaches in terms of efficiency and generalization across building dynamics and operational scenarios remains an open question. In this work, we formally characterize the generalization space for the cross-environment, multi-objective building energy management task, and formulate the multi-objective contextual RL problem. Such a formulation helps understand the challenges of transferring learned policies across varied operational contexts such as climate and heat convection dynamics under multiple control objectives such as comfort level and energy consumption. We provide BEAVER, a principled framework to parameterize such contextual information in realistic building RL environments, and construct a novel benchmark to facilitate the evaluation of generalizable RL algorithms in practical building control tasks.

Contact: yize.chen@ualberta.ca

![BEAVER Framework](https://github.com/chennnnnyize/BEAVER/blob/main/building_1.png)

# Basic Usage
To run our algorithm on Building-dr-3d for a single run:
```bash
python scripts/building-dr-3d.py --cmorl-ipo --num-seeds 1
```

# Reference
We refer to the implementation from [PGMORL](https://github.com/mit-gfx/PGMORL.git) as a base for part of our code.

If you find our paper or code is useful, please consider citing:

@inproceedings{liu2025beaver,
  title={BEAVER: Building Environments with Assessable Variation for Evaluating Multi-Objective Reinforcement Learning},
  author={Liu, Ruohong and Umenberger, Jack and Chen, Yize},
  booktitle={ICML 2025 CO-BUILD Workshop on Computational Optimization of Buildings}
}


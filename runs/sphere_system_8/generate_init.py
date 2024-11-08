#!/usr/bin/env python3

import os

import numpy as np


root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    rng = np.random.default_rng()

    with open(os.path.join(root_dir, 'init.txt'), 'w') as fp:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = -0.5 + i
                    y = -0.5 + j
                    z = -0.5 + k
                    u, v, w = rng.uniform(low=-10.0, high=10.0, size=3)

                    fp.write(f'1.0 0.5 {x:f} {y:f} {z:f} {u:f} {v:f} {w:f}\n')

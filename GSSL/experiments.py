from tqdm import tqdm
import time
import numpy as np

def tqdm_experiment():
    for i in tqdm(range(10), position=0, leave=False):
        for j in tqdm(range(5), position=1, leave=False):
            time.sleep(.1)

def eyesperiment():
    print(np.eye(5)[[3, 1, 0, 1]])

if __name__ == "__main__":
    eyesperiment()
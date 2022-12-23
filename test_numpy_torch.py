import time
import numpy as np
import torch as tr


def main(): 
    a_np = np.random((100,0))
    b_np = np.random((100,0))

    c_np = np.cross(a_np, b_np)
    print(c_np.shape())

    # print(time.time())

if __name__ == "__main__":
    main()

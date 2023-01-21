from tqdm import tqdm
from llio import *

if __name__ == "__main__":

    llio = LLIO("cfg.yml")
    llio.init_integrator()

    print("For each batched datapoint, estimate the state recursively.")
    for idx, data in enumerate(tqdm(llio.dataloader)):
        # main
        state = llio.propogate(data)
        state = llio.correct(data, llio.cfg["use_lidar_correction"])
        llio.append_log(data, state)

        # if you want to see the result's front part ASAP.
        test_num_batches = 10000
        if idx > test_num_batches:
            break

    llio.visualize_traj()

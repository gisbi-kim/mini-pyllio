import os
import sys
import copy
from datetime import datetime
import argparse
from tqdm import tqdm
import tqdm.notebook as tq

import numpy as np
from scipy.spatial.transform import Rotation as R

from llio import *
import utils

if __name__ == "__main__":

    llio = LLIO("cfg.yml")
    llio.init_integrator()

    for idx, data in enumerate(tqdm(llio.dataloader)):

        if idx > 150:
            break

        propagted_state = llio.propogate(data)
        llio.correct(data)

        # if is_true(cfg["visualize_registered_scan"]):
        #     draw_registration_result(
        #         source, target, lidar_delta_tf)

        # # loosely correction
        # if pose_corrected is None:
        #     print(pcd)
        #     pose_corrected = prop_global_pose
        # else:
        #     def update_pose_via_handeye():
        #         return pose_previous @ (calib.velo2imu @ lidar_delta_tf @ calib.imu2velo)

        #     def update_pva(pos, vel, rot):
        #         integrator.pos = pos
        #         integrator.vel = vel
        #         integrator.rot = rot

        #     # see https://pypose.org/docs/main/
        #     # _modules/pypose/module/imu_preintegrator/#IMUPreintegrator
        #     pose_corrected = update_pose_via_handeye()
        #     update_pva(pos=torch.tensor(pose_corrected[:3, -1]).unsqueeze(0),
        #                vel=torch.tensor(
        #                    pose_corrected[:3, -1] - pose_previous[:3, -1]).unsqueeze(0),
        #                rot=pp.SO3(R.from_matrix(pose_corrected[:3, :3]).as_quat()))

        # # renwal for next
        # llio.pose_previous = llio.pose_corrected
        # pcd_previous = pcd
        llio.append_log(data, propagted_state)

    llio.visualize_traj()

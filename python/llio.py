import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.utils.data as Data
import open3d as o3d
import pypose as pp

import utils
from kitti.calib import KittiCalib
from kitti.imu import KittiIMU, imu_collate


class LLIO:
    def __init__(self, cfg_path="cfg.yml"):
        self.debug_msg = False

        self.cfg_path = cfg_path
        self.load_config(self.cfg_path)

        self.load_dataset()

        self.set_caches()

    def set_caches(self):
        self.pose_corrected = None
        self.pcd_previous = None
        self.relative_pose_propagated = None

        self.propagated_state = None

    def load_config(self, cfg_path):
        self.cfg = utils.load_yaml(cfg_path)
        print("A dataset", self.cfg["input"]["dataname"],
              self.cfg["input"]["datadrive"], "is selected.")

        self.gyr_std = self.cfg["gyr_std_const"]
        self.gyr_cov = self.gyr_std**2

        self.acc_std = self.cfg["acc_std_const"]
        self.acc_cov = self.acc_std**2

    def load_dataset(self):
        self.dataset = KittiIMU(
            self.cfg["input"]["dataroot"],
            self.cfg["input"]["dataname"],
            self.cfg["input"]["datadrive"],
            duration=self.cfg["step_size"],
            step_size=self.cfg["step_size"],
        )
        print(f"The sequence has {len(self.dataset)} datapoints.")
        print("Step size is " + str(self.cfg["step_size"]))

        self.dataloader = Data.DataLoader(
            dataset=self.dataset,
            batch_size=1,
            collate_fn=imu_collate,
            shuffle=False
        )

        self.calib = KittiCalib()

    def set_initials(self, initial):
        self.pose_corrected = self.get_SE3(initial)
        self.pcd_previous = utils.downsample_points(
            initial["velodyne"][0], self.cfg)

        self.xyzs = [initial["pos"]]
        self.covs = [torch.zeros(9, 9)]

        self.xyzs_gt = [initial["pos"]]

        print("Iniitial pose is set.")

    def init_integrator(self):

        self.initial = self.dataset.get_init_value()
        self.set_initials(self.initial)

        self.integrator = pp.module.IMUPreintegrator(
            self.initial["pos"], self.initial["rot"], self.initial["vel"],
            gyro_cov=torch.tensor([self.gyr_cov, self.gyr_cov, self.gyr_cov]),
            acc_cov=torch.tensor([self.acc_cov, self.acc_cov, self.acc_cov]),
            prop_cov=True, reset=False
        )
        print("Pypose IMUPreintegrator is generated.")

    def get_rotation(self, data):
        if not utils.is_true(self.cfg['use_groundtruth_rot']):
            # In general, no GT attitude (rotation) could be available.
            pose_previous = self.pose_corrected
            return pose_previous[:3, :3]
        else:
            # This is GT rotation
            return data["init_rot"]

    def propogate(self, data):
        propagated_state = self.integrator(  # == forward()
            dt=data["dt"],
            gyro=data["gyro"],
            acc=data["acc"],
            rot=self.get_rotation(data)
        )
        propagted_pose = self.get_SE3(propagated_state)

        pose_corrected_prev = self.pose_corrected
        self.relative_pose_propagated = \
            np.linalg.inv(pose_corrected_prev) @ propagted_pose

        if not utils.is_true(self.cfg["use_lidar_correction"]):
            # if no lidar aid, correction means identity pass (i.e., propagation only)
            self.update_corrected_pose(propagted_pose)

        self.propagated_state = propagated_state
        return propagated_state

    def correct(self, data, lidar_available=False):
        if lidar_available:
            pcd = utils.downsample_points(data["velodyne"][0], self.cfg)
            dx_imu = self.registration(source=pcd, target=self.pcd_previous)

            pose_corrected_prev = self.pose_corrected
            pose_corrected = pose_corrected_prev @ dx_imu

            dt_batch = data["dt"][..., -1, :] * self.cfg["step_size"]  # sec
            corrected_state = self.update_state(dt_batch, pose_corrected)

            self.update_previous_pcd(pcd)

            return corrected_state
        else:
            return self.propagated_state

    def registration(self, source, target):
        # short names
        v2i = self.calib.velo2imu
        i2v = self.calib.imu2velo

        # because relative_pose_propagated is an imu coordinate.
        dx_imu_initial = self.relative_pose_propagated
        dx_lidar_initial = i2v @ dx_imu_initial @ v2i

        if utils.is_true(self.cfg["lidar_only_mode"]):
            dx_lidar_initial = np.identity(4)

        # register
        icplib = o3d.pipelines.registration
        dx_lidar = icplib.registration_generalized_icp(
            source,
            target,
            self.cfg["icp_inlier_threshold"],
            dx_lidar_initial,
            icplib.TransformationEstimationForGeneralizedICP(),
        ).transformation

        # debug
        def detect_registration_fails():
            error = np.linalg.inv(dx_lidar) @ dx_lidar_initial
            # print("error (xyz):", error[:3, -1].transpose())
            # if the error is big, registration may failed.
            # TODO: some action may be required (e.g., do not update, or kf-based weighting)

        if utils.is_true(self.cfg["visualize_registered_scan"]):
            self.draw_registration(source, target, dx_lidar)

        # result
        dx_imu = v2i @ dx_lidar @ i2v
        return dx_imu

    def update_corrected_pose(self, pose):
        self.pose_corrected = pose

    def update_previous_pcd(self, pcd):
        self.pcd_previous = pcd

    def update_state(self, dt_batch, pose_corrected):
        # prepare
        global_pos = torch.tensor(pose_corrected[:3, -1]).unsqueeze(0)

        pose_corrected_prev = self.pose_corrected
        global_delta_pos = torch.tensor(
            pose_corrected[:3, -1] - pose_corrected_prev[:3, -1]).unsqueeze(0)
        global_vel = global_delta_pos / dt_batch

        global_rot = pp.SO3(R.from_matrix(pose_corrected[:3, :3]).as_quat())

        # update (loosely coupled, thus do explicit replacement) the engine state
        self.integrator.pos = global_pos
        self.integrator.vel = global_vel
        self.integrator.rot = global_rot

        # update the cache
        self.update_corrected_pose(pose_corrected)

        return {"rot": global_rot,
                "vel": global_vel.unsqueeze(1),
                "pos": global_pos.unsqueeze(1),
                "cov": self.propagated_state["cov"]}

    def draw_registration(self, source, target, delta):
        utils.draw_registration_result(source, target, delta)

    def append_log(self, data, state):
        self.xyzs.append(state["pos"][..., -1, :])
        self.covs.append(state["cov"][..., -1, :, :])

        self.xyzs_gt.append(data["gt_pos"][..., -1, :])

    def get_SE3(self, state):
        pose = np.identity(4)  # prop means propagated
        pose[:3, :3] = state['rot'][..., -1, :].matrix().numpy().squeeze()
        pose[:3, -1] = state["pos"][..., -1, :].numpy().squeeze()
        return pose

    def visualize_traj(self):
        utils.visualize({
            "poses": torch.cat(self.xyzs).numpy(),
            "poses_gt": torch.cat(self.xyzs_gt).numpy(),
            "covs": torch.stack(self.covs, dim=0).numpy(),
            "cfg": self.cfg,
        })

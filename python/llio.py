import copy
import numpy as np
import torch
import torch.utils.data as Data
import open3d as o3d
import pypose as pp

import utils
from kitti.calib import KittiCalib
from kitti.imu import KittiIMU, imu_collate

import ipdb
# usage: ipdb.set_trace()


class LLIO:
    def __init__(self, cfg_path="cfg.yml"):
        self.cfg_path = cfg_path
        self.load_config(self.cfg_path)

        self.load_dataset()

        self.set_caches()

    def set_caches(self):
        self.pose_previous = None
        self.pose_corrected = None

        self.pcd_previous = None

        self.relative_pose = None

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
        self.pose_previous = self.get_SE3(initial)
        self.pose_corrected = copy.deepcopy(self.pose_previous)
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
        if not utils.is_true(self.cfg['use_rot_initial']):
            # In general, no GT attitude (rotation) could be available.
            if self.pose_previous is not None:
                # this is lidar-aided rotation
                return self.pose_previous[:3, :3]
            else:
                # if a no explicit rotation is provided (i.e., None), use the internally maintained rotation (see https://abit.ly/3acvti and https://abit.ly/9iliex)
                return None
        else:
            # This is GT rotation
            return data["init_rot"]

    def propogate(self, data):
        propagted_state = self.integrator(  # == forward()
            dt=data["dt"],
            gyro=data["gyro"],
            acc=data["acc"],
            rot=self.get_rotation(data)
        )
        propagted_pose = self.get_SE3(propagted_state)
        self.relative_pose = np.linalg.inv(self.pose_previous) @ propagted_pose

        self.update_previous_pose(propagted_pose)

        return propagted_state

    def correct(self, data):
        pcd = utils.downsample_points(data["velodyne"][0], self.cfg)

        self.registration(source=pcd, target=self.pcd_previous)

    def registration(self, source, target):
        # We would use point2plane loss and the point2plane loss required normals
        source.estimate_normals()
        target.estimate_normals()

        # because relative_pose is an imu coordinate.
        dx_initial = self.calib.imu2velo @ self.relative_pose @ self.calib.velo2imu

        # register
        icplib = o3d.pipelines.registration
        dx = icplib.registration_icp(
            source,
            target,
            self.cfg["icp_inlier_threshold"],
            dx_initial,
            icplib.TransformationEstimationPointToPlane(),
            icplib.ICPConvergenceCriteria(max_iteration=30)
        ).transformation

        # result
        diff = np.linalg.inv(dx) @ dx_initial
        print("diff ", diff[:3, -1].transpose())

    def update_previous_pose(self, pose):
        self.pose_previous = pose

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

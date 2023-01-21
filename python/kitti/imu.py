from datetime import datetime

import numpy as np
import torch
import torch.utils.data as Data

import ipdb
import pykitti
import pypose as pp


class KittiIMU(Data.Dataset):
    def __init__(self, root, dataname, drive, duration=2, step_size=1):
        super().__init__()
        self.duration = duration
        self.data = pykitti.raw(root, dataname, drive)
        self.seq_len = len(self.data.timestamps) - 1

        self.dt = torch.tensor(
            [
                datetime.timestamp(self.data.timestamps[i + 1])
                - datetime.timestamp(self.data.timestamps[i])
                for i in range(self.seq_len)
            ]
        )

        print("Loading oxts (i.e., IMU and GT pose) data ...")
        self.gyro = torch.tensor(
            [
                [
                    self.data.oxts[i].packet.wx,
                    self.data.oxts[i].packet.wy,
                    self.data.oxts[i].packet.wz,
                ]
                for i in range(self.seq_len)
            ]
        )
        self.acc = torch.tensor(
            [
                [
                    self.data.oxts[i].packet.ax,
                    self.data.oxts[i].packet.ay,
                    self.data.oxts[i].packet.az,
                ]
                for i in range(self.seq_len)
            ]
        )
        self.gt_rot = pp.euler2SO3(
            torch.tensor(
                [
                    [
                        self.data.oxts[i].packet.roll,
                        self.data.oxts[i].packet.pitch,
                        self.data.oxts[i].packet.yaw,
                    ]
                    for i in range(self.seq_len)
                ]
            )
        )
        self.gt_vel = self.gt_rot @ torch.tensor(
            [
                [
                    self.data.oxts[i].packet.vf,
                    self.data.oxts[i].packet.vl,
                    self.data.oxts[i].packet.vu,
                ]
                for i in range(self.seq_len)
            ]
        )
        self.gt_pos = torch.tensor(
            np.array([self.data.oxts[i].T_w_imu[0:3, 3]
                     for i in range(self.seq_len)])
        )

        print(
            f"Loading {self.seq_len} velodyne data (may require a few GB memory) ...")
        self.velodyne = [(self.data.get_velo(i)) for i in range(self.seq_len)]
        print("In this sequence, the number of (unskipped) point cloud scans:", len(
            self.velodyne))
        print("An example scan has", self.velodyne[6].shape, "points.")

        start_frame = 0
        end_frame = self.seq_len

        self.index_map = [
            i for i in range(0, end_frame - start_frame - self.duration, step_size)
        ]
        # print(f"self.index_map is {self.index_map}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        return {
            "dt": self.dt[frame_id:end_frame_id],
            "acc": self.acc[frame_id:end_frame_id],
            "gyro": self.gyro[frame_id:end_frame_id],
            "gt_pos": self.gt_pos[frame_id + 1: end_frame_id + 1],
            "gt_rot": self.gt_rot[frame_id + 1: end_frame_id + 1],
            "gt_vel": self.gt_vel[frame_id + 1: end_frame_id + 1],
            "velodyne": self.velodyne[frame_id],
            "init_pos": self.gt_pos[frame_id][None, ...],
            # TODO: the init rotation might be used in gravity compensation
            "init_rot": self.gt_rot[frame_id:end_frame_id],
            "init_vel": self.gt_vel[frame_id][None, ...],
        }

    def get_init_value(self):
        return {"pos": self.gt_pos[:1],
                "rot": self.gt_rot[:1],
                "vel": self.gt_vel[:1],
                "velodyne": self.velodyne[:1]}


def imu_collate(data):
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    gt_rot = torch.stack([d["gt_rot"] for d in data])
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    init_rot = torch.stack([d["init_rot"] for d in data])
    init_vel = torch.stack([d["init_vel"] for d in data])

    dt = torch.stack([d["dt"] for d in data]).unsqueeze(-1)

    velodyne = [d["velodyne"] for d in data]

    return {
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "gt_rot": gt_rot,
        "velodyne": velodyne,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_rot": init_rot,
    }

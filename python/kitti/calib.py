import numpy as np


class KittiCalib:
    def __init__(self):
        # 2011_09_30
        imu2velo_rot = np.array([9.999976e-01, 7.553071e-04, -2.035826e-03,
                                -7.854027e-04, 9.998898e-01, -1.482298e-02,
                                2.024406e-03, 1.482454e-02, 9.998881e-01]).reshape(3, 3)
        imu2velo_trans = np.array(
            [-8.086759e-01, 3.195559e-01, -7.997231e-01])

        imu2velo = np.identity(4)
        imu2velo[:3, :3] = imu2velo_rot
        imu2velo[:3, -1] = imu2velo_trans

        velo2imu = np.linalg.inv(imu2velo)

        self.imu2velo = imu2velo
        self.velo2imu = velo2imu

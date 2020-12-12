ROOT_DIR = os.path.abspath("/Dataset")
class IMUDataset(Dataset):
    
    def __init__(self, dataset, transform=None):
        self.imu = pd.read_csv(ROOT_DIR + dataset + "/mav0/imu0/data.csv").values
        self.pose = pd.read_csv(ROOT_DIR + dataset + "/mav0/state_groundtruth_estimate0/data.csv").values
        self.interpolate()
        print("IMU shape",self.imu.shape)
        print("Pose shape", self.pose.shape)
        self.window = 200
        self.stride = 10
        print("Im in constructor")
        self.initial_pose = self.pose[0, 1:8]
        
        self.i = 0
    def hamilton_product(self, quat1, quat2):
        a1, b1, c1, d1 = quat1
        a2, b2, c2, d2 = quat2
        q1 = a1*a2- b1*b2 - c1*c2 -d1*d2
        q2 = a1*b2 + b1*a2 + c1*d2 - d1*c2
        q3 = a1*c2 - b1*d2 + c1*a2 + d1*b2
        q4 = a1*d2 + b1*c2 - c1*b2 + d1*a2
        return np.array([q1, q2, q3, q4])

    def __len__(self):
        print("The length is")
        if len(self.imu) < len(self.pose):
            return (len(self.imu) - self.window) // 10
        else:
            return (len(self.pose) - self.window)  // 10
            
    """def __len__(self):
        return (len(self.imu) - self.window) // self.stride"""

    def interpolate(self):
        imu_timestamp = self.imu[:, 0]
        pose_timestamp = self.pose[:, 0]
        func = interpolate.interp1d(imu_timestamp, self.imu[:, 1:7], axis=0)
        self.imu = func(pose_timestamp)
        assert(self.imu.shape[0] == self.pose.shape[0])
    
    def gen_relative_pose(self, pose):
        length = pose.shape[0]
        pose_a = pose[length//2 - self.stride//2, :]
        pose_b = pose[length//2 + self.stride//2, :]

        quat_a = pose_a[3:]
        quat_b = pose_b[3:]
        """Rotation matrix of a quaternion"""
        r = R.from_quat(quat_a) 
        q_mat = r.as_matrix().T
        """Conjugate of a quaternion"""
        q_conj = quat_a * [1 , -1, -1 , -1] 
        delta_position = np.matmul(q_mat, (pose_a[:3] - pose_b[:3]))
        delta_quat = self.hamilton_product(q_conj, quat_b)
        #return np.concatenate((delta_position, self.force_quaternion_uniqueness(delta_quat)))
        return np.concatenate((delta_position, delta_quat))

    def absolute_pose_estimation(self, delta_p, delta_q):
        pt_1 = self.initial_pose[:3]
        qt_1 = self.initial_pose[3:]
        r = R.from_quat(qt_1)
        qt_1_mat = r.as_matrix()
        p = pt_1 +  np.matmul(qt_1_mat, delta_p)
        q = self.hamilton_product(qt_1, delta_q)
        self.initial_pose = np.concatenate((p, q))
        return self.initial_pose

    def force_quaternion_uniqueness(self, q):
        if np.absolute(q[0]) > 1e-05:
            if q[0] < 0:
                return -q
            else:
                return q
        elif np.absolute(q[1]) > 1e-05:
            if q[1] < 0:
                return -q
            else:
                return q
        elif np.absolute(q[2]) > 1e-05:
            if q[2] < 0:
                return -q
            else:
                return q
        else:
            if q[3] < 0:
                return -q
            else:
                return q

    def __getitem__(self, idx):
        imu = self.imu[10 * idx : 10 * idx + 200 , :]
        pose = self.pose[10 * idx : 10 * idx + 200 , 1:8]
        
        relative_pose = self.gen_relative_pose(pose)
        absolute_pose_estimate = self.absolute_pose_estimation(relative_pose[:3], relative_pose[3:])
       
        return {"imu": imu.reshape((-1, 200)) , "relative_pose": relative_pose, "absolute_pose_estimate": absolute_pose_estimate}
        

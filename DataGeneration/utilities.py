import numpy as np
import torch 
from scipy.spatial.transform import Rotation as R
# import rclpy
# from rclpy.node import Node
from std_msgs.msg import Header
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
import math
import pdb


class Utils:
    def __init__(self, K):
        self.K = K
        self.pose_dict = {
            'x': 0,
            'y': 1,
            'z': 2,
            'roll': 3,
            'pitch': 4,
            'yaw': 5
        }

    def sequence_transform_to_world_torch(self, current_pose, del_pose_seq):
        """
        Transform a sequence of relative delta poses into absolute world poses:

        Args: 
            current pose: torch.Tensor: (K, 6) or (6,) - starting pose [x, y, z, roll, pitch, yaw]
            del_pose_seq torch.tensor (K, N, 6) - predicted delta poses
        
        Returns:
            world poses torch.Tensor: (K, N, 6) absolute poses along the horizon
        """
        # Ensure del_pose_seq is (K, N, 6)
        if del_pose_seq.dim() == 2:
            del_pose_seq = del_pose_seq.unsqueeze(0)

        K, N, _ = del_pose_seq.shape
        device = current_pose.device
        dtype = current_pose.dtype

        # if unmatching dim, unsqueeze
        if current_pose.dim() == 1:
            current_pose = current_pose.unsqueeze(0).repeat(K, 1)

        # Variable for holding the final world poses
        world_poses = torch.zeros((K, N, 6), device=device, dtype=dtype)

        # current state disected
        curr_pos = current_pose[:, :3].clone()
        curr_angles = current_pose[:, 3:].clone()
    
        for i in range(N):
            delta = del_pose_seq[:, i, :]


            current_pose = to_world_torch(current_pose, delta)
            # current_pose = world_pose
            world_poses[:, i, :] = current_pose.clone()
            # Build current rotation matrices (ZYX order)
            # current_yaw_pitch_roll = torch.stack([curr_angles[:, 2], curr_angles[:, 1], curr_angles[:, 0]], dim=1)
            # rot_mats = self._euler_to_rotation_matrix_torch(current_yaw_pitch_roll)

            # # Local Translation -> world translation
            # dpos_local = delta[:, :3]
            # # R @ dpos_local 
            # dpos_local = delta[:, :3]
            # dpos_world = torch.bmm(rot_mats, dpos_local.unsqueeze(2)).squeeze(2)

            # #update position
            # new_pos = curr_pos + dpos_world

            # #update angles
            # new_angles = curr_angles + delta[:, 3:]

            # # Store
            # world_poses[:, i, :3] = new_pos
            # world_poses[:, i, 3:] = new_angles 

            # #propagate for next step
            # curr_pos = new_pos
            # curr_angles = new_angles 

        return world_poses, world_poses[:, -1, :]

    def normalize_standard(self, data, mean, std, state_dim=6):
        # if state_dim != 6:
        #     return (data - mean.unsqueeze(0).unsqueeze(0)[:,:,[0, 1, 5]]) / (std.unsqueeze(0).unsqueeze(0)[:,:,[0, 1, 5]] + 1e-8)
        return (data - mean)/ (std + 1e-8)

    def un_unormalize_standard(self, data, mean, std, state_dim=6):
        if state_dim != 6:
            return (data * (std.unsqueeze(0).unsqueeze(0)[:,:,[0, 1, 5]] + 1e-8) ) + mean.unsqueeze(0).unsqueeze(0)[:,:,[0, 1, 5]]
        return (data * (std + 1e-8) ) + mean
    
    @staticmethod
    def _euler_to_rotation_matrix_torch(euler_angles):
        """ Convert Euler angles (yaw, pitch, roll) to a rotation matrix batch (ZYX convention) """
        # euler_angles shape: (Batch, 3) where columns are [yaw, pitch, roll]
        
        yaw = euler_angles[:, 0]
        pitch = euler_angles[:, 1]
        roll = euler_angles[:, 2]
        
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cr, sr = torch.cos(roll), torch.sin(roll)
        
        # ZYX Euler angles to Rotation Matrix
        R_z = torch.stack([
            torch.stack([cy, -sy, torch.zeros_like(cy)], dim=1),
            torch.stack([sy, cy, torch.zeros_like(cy)], dim=1),
            torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)], dim=1)
        ], dim=1)
        
        R_y = torch.stack([
            torch.stack([cp, torch.zeros_like(cp), sp], dim=1),
            torch.stack([torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp)], dim=1),
            torch.stack([-sp, torch.zeros_like(cp), cp], dim=1)
        ], dim=1)

        R_x = torch.stack([
            torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)], dim=1),
            torch.stack([torch.zeros_like(cr), cr, -sr], dim=1),
            torch.stack([torch.zeros_like(cr), sr, cr], dim=1)
        ], dim=1)

        # R = Rz * Ry * Rx
        R = torch.bmm(R_z, torch.bmm(R_y, R_x))
        return R # (K, 3, 3)
    
    def to_robot_torch(self, Robot_frame, P_relative):
        SE3 = True

        if not isinstance(Robot_frame, torch.Tensor):
            Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)
        
        if not isinstance(P_relative, torch.Tensor):
            P_relative = torch.tensor(P_relative, dtype=torch.float32)

        if len(Robot_frame.shape) == 1:
            Robot_frame = Robot_frame.unsqueeze(0)

        if len(P_relative.shape) == 1:
            P_relative = P_relative.unsqueeze(0)
    
        if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
            raise ValueError(f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}")
        
        if Robot_frame.shape != P_relative.shape:
            raise ValueError("Input tensors must have same shape")
        
        if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
            raise ValueError(f"Input tensors must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}")
        
        if Robot_frame.shape[-1] == 3:
            SE3 = False
            Robot_frame_ = torch.zeros((Robot_frame.shape[0], 6), device=Robot_frame.device, dtype=Robot_frame.dtype)
            Robot_frame_[:, [0,1,5]] = Robot_frame
            Robot_frame = Robot_frame_
            P_relative_ = torch.zeros((P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype)
            P_relative_[:, [0,1,5]] = P_relative
            P_relative = P_relative_
            
        """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
        batch_size = Robot_frame.shape[0]
        ones = torch.ones_like(P_relative[:, 0])
        transform = torch.zeros_like(Robot_frame)
        T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
        T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

        T1[:, :3, :3] = self._euler_to_rotation_matrix_torch(Robot_frame[:, 3:])
        T2[:, :3, :3] = self._euler_to_rotation_matrix_torch(P_relative[:, 3:])
        T1[:, :3,  3] = Robot_frame[:, :3]
        T2[:, :3,  3] = P_relative[:, :3]
        T1[:,  3,  3] = 1
        T2[:,  3,  3] = 1 
        
        T1_inv = torch.inverse(T1)
        tf3_mat = torch.matmul(T2, T1_inv)
        
        transform[:, :3] = torch.matmul(T1_inv, torch.cat((P_relative[:,:3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze(dim=2)[:, :3]
        transform[:, 3:] = self.extract_euler_angles_from_se3_batch(tf3_mat)
        
        if not SE3:
            transform = transform[:, [0,1,5]]
        
        return transform
    
    def extract_euler_angles_from_se3_batch(self,tf3_matx):
        # Validate input shape
        if tf3_matx.shape[1:] != (4, 4):
            raise ValueError("Input tensor must have shape (batch, 4, 4)")

        # Extract rotation matrices
        rotation_matrices = tf3_matx[:, :3, :3]

        # Initialize tensor to hold Euler angles
        batch_size = tf3_matx.shape[0]
        euler_angles = torch.zeros((batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype)

        # Compute Euler angles
        euler_angles[:, 0] = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
        euler_angles[:, 1] = torch.atan2(-rotation_matrices[:, 2, 0], torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
        euler_angles[:, 2] = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw

        return euler_angles
    
    # Convert roll, pitch, and yaw angles to quaternion
    def rpy_to_quaternion(self, roll, pitch, yaw):
        quaternion = Quaternion()
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        quaternion.x = sr * cp * cy - cr * sp * sy
        quaternion.y = cr * sp * cy + sr * cp * sy
        quaternion.z = cr * cp * sy - sr * sp * cy
        quaternion.w = cr * cp * cy + sr * sp * sy

        return quaternion
    
    # Convert quaternion to roll, pitch, and yaw angles
    def quaternion_to_rpy(self, quaternion):
        qw = quaternion.w
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
    def make_header(self, node, frame_id, stamp=None):
        if stamp == None:
            stamp = node.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header
    
    def particle_to_posestamped(self, node, particle, frame_id):
        pose = PoseStamped()
        pose.header = self.make_header(node, frame_id)
        if particle.shape [0] == 3:
            particle_ = torch.zeros((6,), device=particle.device, dtype=particle.dtype)
            particle_[[0,1,5]] = particle
            particle = particle_

        pose.pose.position.x = particle[0].item()
        pose.pose.position.y = particle[1].item()
        pose.pose.position.z = particle[2].item()
        pose.pose.orientation = self.rpy_to_quaternion(particle[3].item(), particle[4].item(), particle[5].item())
        return pose

    def make_header_dict(self, node, frame_id, stamp=None):
        """
        Creates a std_msgs/Header dictionary.
        """
        if stamp is None:
            # pdb.set_trace()
            # Generate current time in ROS format (secs/nsecs)
            now = node.get_clock().now().to_msg()
            # secs = now.sec 
            # nsecs = now.nanosec 
            stamp = {'secs': now.sec, 'nsecs': now.nanosec}
            
        return {
            'frame_id': frame_id,
            'stamp': stamp
        }
    
    def particle_to_dict_posestamped(self, node, particle, frame_id):
        if particle.shape [0] == 3:
            particle_full = torch.zeros((6,), device=particle.device, dtype=particle.dtype)
            particle_full[0] = particle[0]
            particle_full[1] = particle[1]
            particle_full[5] = particle[2]
        else:
            particle_full = particle

        quat = self.rpy_to_quaternion(particle_full[3].item(), particle_full[4].item(), particle_full[5].item())  
        pose_msg = {
            'header': self.make_header_dict(node, frame_id),
            'pose': {
                'position': {
                    'x': particle_full[0].item(),
                    'y': particle_full[1].item(),
                    'z': particle_full[2].item()
                },
                'orientation': {
                    'x': quat.x,
                    'y': quat.y,
                    'z': quat.z,
                    'w': quat.w
                }
            }
        }


        return pose_msg


def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles to a rotation matrix """
    # Compute sin and cos for Euler angles
    cos = torch.cos(euler_angles)
    sin = torch.sin(euler_angles)
    zero = torch.zeros_like(euler_angles[:, 0])
    one = torch.ones_like(euler_angles[:, 0])
    # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
    R_x = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], dim=1).view(-1, 3, 3)
    R_y = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], dim=1).view(-1, 3, 3)
    R_z = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], dim=1).view(-1, 3, 3)

    return torch.matmul(torch.matmul(R_z, R_y), R_x)

def extract_euler_angles_from_se3_batch(tf3_matx):
    # Validate input shape
    if tf3_matx.shape[1:] != (4, 4):
        raise ValueError("Input tensor must have shape (batch, 4, 4)")

    # Extract rotation matrices
    rotation_matrices = tf3_matx[:, :3, :3]

    # Initialize tensor to hold Euler angles
    batch_size = tf3_matx.shape[0]
    euler_angles = torch.zeros((batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype)

    # Compute Euler angles
    euler_angles[:, 0] = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
    euler_angles[:, 1] = torch.atan2(-rotation_matrices[:, 2, 0], torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
    euler_angles[:, 2] = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw

    return euler_angles

def to_robot_torch(Robot_frame, P_relative):
    SE3 = True

    if not isinstance(Robot_frame, torch.Tensor):
        Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)
    
    if not isinstance(P_relative, torch.Tensor):
        P_relative = torch.tensor(P_relative, dtype=torch.float32)

    if len(Robot_frame.shape) == 1:
        Robot_frame = Robot_frame.unsqueeze(0)

    if len(P_relative.shape) == 1:
        P_relative = P_relative.unsqueeze(0)
    
    if Robot_frame.device != P_relative.device:
        P_relative = P_relative.to(Robot_frame.device)

    if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
        raise ValueError(f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}")
    
    if Robot_frame.shape != P_relative.shape:
        raise ValueError("Input tensors must have same shape")
    
    if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
        raise ValueError(f"Input tensors must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}")
    
    if Robot_frame.shape[-1] == 3:
        SE3 = False
        Robot_frame_ = torch.zeros((Robot_frame.shape[0], 6), device=Robot_frame.device, dtype=Robot_frame.dtype)
        Robot_frame_[:, [0,1,5]] = Robot_frame
        Robot_frame = Robot_frame_
        P_relative_ = torch.zeros((P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype)
        P_relative_[:, [0,1,5]] = P_relative
        P_relative = P_relative_
        
    """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
    batch_size = Robot_frame.shape[0]
    ones = torch.ones_like(P_relative[:, 0])
    transform = torch.zeros_like(Robot_frame)
    T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
    T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

    T1[:, :3, :3] = euler_to_rotation_matrix(Robot_frame[:, 3:])
    T2[:, :3, :3] = euler_to_rotation_matrix(P_relative[:, 3:])
    T1[:, :3,  3] = Robot_frame[:, :3]
    T2[:, :3,  3] = P_relative[:, :3]
    T1[:,  3,  3] = 1
    T2[:,  3,  3] = 1 
    
    T1_inv = torch.inverse(T1)
    tf3_mat = torch.matmul(T1_inv, T2)
    
    # transform[:, :3] = torch.matmul(T1_inv, torch.cat((P_relative[:,:3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze(dim=2)[:, :3]
    transform[:, :3] = tf3_mat[:, :3, 3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(tf3_mat)
    
    if not SE3:
        transform = transform[:, [0,1,5]]
    
    return transform

def to_world_torch(Robot_frame, P_relative):
    SE3 = True

    if not isinstance(Robot_frame, torch.Tensor):
        Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)
    if not isinstance(P_relative, torch.Tensor):
        P_relative = torch.tensor(P_relative, dtype=torch.float32)

    if len(Robot_frame.shape) == 1:
        Robot_frame = Robot_frame.unsqueeze(0)

    if len(P_relative.shape) == 1:
        P_relative = P_relative.unsqueeze(0)

    if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
        raise ValueError(f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}")

    if Robot_frame.device != P_relative.device:
        P_relative = P_relative.to(Robot_frame.device)

    if Robot_frame.shape != P_relative.shape:
        raise ValueError("Input tensors must have same shape")
    
    if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
        raise ValueError(f"Input tensors 1 must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}")
    
    if P_relative.shape[-1] != 6 and P_relative.shape[-1] != 3:
        raise ValueError(f"Input tensors 2 must have last dim equal to 6 for SE3 and 3 for SE2 got {P_relative.shape[-1]}")
    
    if Robot_frame.shape[-1] == 3:
        SE3 = False
        Robot_frame_ = torch.zeros((Robot_frame.shape[0], 6), device=Robot_frame.device, dtype=Robot_frame.dtype)
        Robot_frame_[:, [0,1,5]] = Robot_frame
        Robot_frame = Robot_frame_
        P_relative_ = torch.zeros((P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype)
        P_relative_[:, [0,1,5]] = P_relative
        P_relative = P_relative_
        
    """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
    batch_size = Robot_frame.shape[0]
    ones = torch.ones_like(P_relative[:, 0])
    transform = torch.zeros_like(Robot_frame)
    T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
    T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

    R1 = euler_to_rotation_matrix(Robot_frame[:, 3:])
    R2 = euler_to_rotation_matrix(P_relative[:, 3:])
    
    T1[:, :3, :3] = R1
    T2[:, :3, :3] = R2
    T1[:, :3,  3] = Robot_frame[:, :3]
    T2[:, :3,  3] = P_relative[:, :3]
    T1[:,  3,  3] = 1
    T2[:,  3,  3] = 1 

    T_tf = torch.matmul(T1, T2)
    # transform[:, :3] = torch.matmul(T1, torch.cat((P_relative[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze(dim=2)[:, :3]
    transform[:, :3] = T_tf[:, :3, 3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(T_tf)

    if not SE3:
        transform = transform[:, [0,1,5]]

    return transform
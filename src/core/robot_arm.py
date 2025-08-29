"""
Core robot arm implementation for PyBullet simulation.
"""

import pybullet as p
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotArm:
    """Base robot arm class with complete kinematics and control functionality."""
    
    def __init__(self, urdf_path: str, base_position: List[float] = [0, 0, 0]):
        """
        Initialize robot arm.
        
        Args:
            urdf_path: Path to robot URDF file
            base_position: Base position in world coordinates [x, y, z]
        """
        self.urdf_path = urdf_path
        self.base_position = base_position
        self.robot_id = None
        self.num_joints = 0
        self.joint_indices = []
        self.gripper_indices = []
        self.end_effector_index = 6  # Default for KUKA iiwa
        self.joint_limits = []
        self.home_position = [0, 0, 0, -1.57, 0, 1.57, 0]  # Default home position
        self.current_joint_positions = []
        self.gripper_open = True
        
    def load_robot(self, physics_client_id: int = 0) -> int:
        """
        Load robot into PyBullet simulation.
        
        Args:
            physics_client_id: PyBullet physics client ID
            
        Returns:
            Robot ID in simulation
        """
        try:
            self.robot_id = p.loadURDF(
                self.urdf_path, 
                basePosition=self.base_position,
                physicsClientId=physics_client_id
            )
            
            # Get joint information
            self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=physics_client_id)
            self.joint_indices = []
            self.gripper_indices = []
            self.joint_limits = []
            
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=physics_client_id)
                joint_name = joint_info[1].decode('utf-8')
                joint_type = joint_info[2]
                
                # Add revolute and prismatic joints
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    if 'gripper' in joint_name.lower() or 'finger' in joint_name.lower():
                        self.gripper_indices.append(i)
                    else:
                        self.joint_indices.append(i)
                        
                    # Get joint limits
                    lower_limit = joint_info[8]
                    upper_limit = joint_info[9]
                    self.joint_limits.append((lower_limit, upper_limit))
            
            # Initialize joint positions
            self.current_joint_positions = [0.0] * len(self.joint_indices)
            
            logger.info(f"Robot loaded with ID: {self.robot_id}")
            logger.info(f"Controllable joints: {len(self.joint_indices)}")
            logger.info(f"Gripper joints: {len(self.gripper_indices)}")
            
            return self.robot_id
            
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            raise
    
    def get_joint_positions(self) -> List[float]:
        """
        Get current joint positions.
        
        Returns:
            List of joint angles in radians
        """
        if self.robot_id is None:
            raise ValueError("Robot not loaded")
            
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        self.current_joint_positions = [state[0] for state in joint_states]
        return self.current_joint_positions.copy()
    
    def set_joint_positions(self, positions: List[float], max_force: float = 500.0) -> None:
        """
        Set target joint positions.
        
        Args:
            positions: Target joint angles in radians
            max_force: Maximum force for joint control
        """
        if self.robot_id is None:
            raise ValueError("Robot not loaded")
            
        if len(positions) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} positions, got {len(positions)}")
        
        # Clamp positions to joint limits
        clamped_positions = []
        for i, pos in enumerate(positions):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if lower < upper:  # Valid limits
                    pos = max(lower, min(upper, pos))
            clamped_positions.append(pos)
        
        # Set joint positions
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=clamped_positions,
            forces=[max_force] * len(self.joint_indices)
        )
        
        self.current_joint_positions = clamped_positions
    
    def get_end_effector_position(self) -> Tuple[float, float, float]:
        """
        Get current end-effector position.
        
        Returns:
            End-effector position (x, y, z)
        """
        if self.robot_id is None:
            raise ValueError("Robot not loaded")
            
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        position = link_state[0]  # World position
        return position
    
    def get_end_effector_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Get current end-effector pose (position and orientation).
        
        Returns:
            Tuple of (position, orientation_quaternion)
        """
        if self.robot_id is None:
            raise ValueError("Robot not loaded")
            
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        position = link_state[0]
        orientation = link_state[1]
        return position, orientation
    
    def move_to_position(self, target_pos: List[float], orientation: Optional[List[float]] = None) -> bool:
        """
        Move end-effector to target position using inverse kinematics.
        
        Args:
            target_pos: Target position [x, y, z]
            orientation: Target orientation quaternion [x, y, z, w] (optional)
            
        Returns:
            True if movement successful, False otherwise
        """
        try:
            joint_positions = self.inverse_kinematics(target_pos, orientation)
            if joint_positions:
                self.set_joint_positions(joint_positions)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to move to position {target_pos}: {e}")
            return False
    
    def inverse_kinematics(self, target_pos: List[float], orientation: Optional[List[float]] = None) -> List[float]:
        """
        Calculate inverse kinematics for target position.
        
        Args:
            target_pos: Target position [x, y, z]
            orientation: Target orientation quaternion [x, y, z, w] (optional)
            
        Returns:
            Joint angles for target pose
        """
        if self.robot_id is None:
            raise ValueError("Robot not loaded")
        
        # Use current joint positions as seed
        current_joints = self.get_joint_positions()
        
        if orientation is None:
            # Calculate IK for position only
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                target_pos,
                lowerLimits=[limit[0] for limit in self.joint_limits[:len(self.joint_indices)]],
                upperLimits=[limit[1] for limit in self.joint_limits[:len(self.joint_indices)]],
                jointRanges=[(limit[1] - limit[0]) for limit in self.joint_limits[:len(self.joint_indices)]],
                restPoses=current_joints
            )
        else:
            # Calculate IK for position and orientation
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                target_pos,
                orientation,
                lowerLimits=[limit[0] for limit in self.joint_limits[:len(self.joint_indices)]],
                upperLimits=[limit[1] for limit in self.joint_limits[:len(self.joint_indices)]],
                jointRanges=[(limit[1] - limit[0]) for limit in self.joint_limits[:len(self.joint_indices)]],
                restPoses=current_joints
            )
        
        # Return only the controllable joints
        return list(joint_positions[:len(self.joint_indices)])
    
    def forward_kinematics(self, joint_angles: List[float]) -> List[float]:
        """
        Calculate forward kinematics for given joint angles.
        
        Args:
            joint_angles: Joint angles in radians
            
        Returns:
            End-effector position [x, y, z]
        """
        if self.robot_id is None:
            raise ValueError("Robot not loaded")
            
        # Temporarily set joint positions to calculate FK
        current_positions = self.get_joint_positions()
        self.set_joint_positions(joint_angles)
        
        # Get end-effector position
        position = self.get_end_effector_position()
        
        # Restore original positions
        self.set_joint_positions(current_positions)
        
        return list(position)
    
    def reset_to_home(self) -> None:
        """Reset robot to home position."""
        if len(self.home_position) == len(self.joint_indices):
            self.set_joint_positions(self.home_position)
            logger.info("Robot reset to home position")
        else:
            logger.warning("Home position not properly configured")
    
    def enable_torque_control(self) -> None:
        """Enable torque control mode for all joints."""
        if self.robot_id is None:
            raise ValueError("Robot not loaded")
            
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.TORQUE_CONTROL,
            forces=[0] * len(self.joint_indices)
        )
        logger.info("Torque control enabled")
    
    def get_gripper_state(self) -> bool:
        """
        Get current gripper state.
        
        Returns:
            True if gripper is open, False if closed
        """
        return self.gripper_open
    
    def open_gripper(self, max_force: float = 50.0) -> None:
        """
        Open the gripper.
        
        Args:
            max_force: Maximum force for gripper control
        """
        if self.robot_id is None or not self.gripper_indices:
            logger.warning("No gripper available")
            return
            
        # Set gripper joints to open position
        open_positions = [0.04] * len(self.gripper_indices)  # Typical open position
        
        p.setJointMotorControlArray(
            self.robot_id,
            self.gripper_indices,
            p.POSITION_CONTROL,
            targetPositions=open_positions,
            forces=[max_force] * len(self.gripper_indices)
        )
        
        self.gripper_open = True
        logger.info("Gripper opened")
    
    def close_gripper(self, max_force: float = 50.0) -> None:
        """
        Close the gripper.
        
        Args:
            max_force: Maximum force for gripper control
        """
        if self.robot_id is None or not self.gripper_indices:
            logger.warning("No gripper available")
            return
            
        # Set gripper joints to closed position
        closed_positions = [0.0] * len(self.gripper_indices)
        
        p.setJointMotorControlArray(
            self.robot_id,
            self.gripper_indices,
            p.POSITION_CONTROL,
            targetPositions=closed_positions,
            forces=[max_force] * len(self.gripper_indices)
        )
        
        self.gripper_open = False
        logger.info("Gripper closed")
    
    def get_joint_info(self) -> dict:
        """
        Get comprehensive joint information.
        
        Returns:
            Dictionary with joint information
        """
        if self.robot_id is None:
            raise ValueError("Robot not loaded")
            
        info = {
            'num_joints': self.num_joints,
            'controllable_joints': len(self.joint_indices),
            'gripper_joints': len(self.gripper_indices),
            'joint_limits': self.joint_limits,
            'current_positions': self.get_joint_positions(),
            'end_effector_position': self.get_end_effector_position(),
            'gripper_open': self.gripper_open
        }
        
        return info
    
    def is_position_reachable(self, target_pos: List[float]) -> bool:
        """
        Check if a target position is reachable.
        
        Args:
            target_pos: Target position [x, y, z]
            
        Returns:
            True if position is reachable, False otherwise
        """
        try:
            joint_positions = self.inverse_kinematics(target_pos)
            
            # Check if IK solution is valid
            if joint_positions:
                # Verify the solution by forward kinematics
                calculated_pos = self.forward_kinematics(joint_positions)
                error = np.linalg.norm(np.array(target_pos) - np.array(calculated_pos))
                return error < 0.01  # 1cm tolerance
            
            return False
        except Exception:
            return False 
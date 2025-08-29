"""
KUKA iiwa specific robot implementation.
"""

import os
from typing import List
import logging
from ..core.robot_arm import RobotArm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KukaIiwa(RobotArm):
    """KUKA iiwa 7-DOF robot arm implementation."""
    
    # KUKA iiwa joint limits (radians)
    JOINT_LIMITS = [
        (-2.97, 2.97),    # Joint 1
        (-2.09, 2.09),    # Joint 2
        (-2.97, 2.97),    # Joint 3
        (-2.09, 2.09),    # Joint 4
        (-2.97, 2.97),    # Joint 5
        (-2.09, 2.09),    # Joint 6
        (-3.05, 3.05)     # Joint 7
    ]
    
    # Default home position
    HOME_POSITION = [0, 0, 0, -1.57, 0, 1.57, 0]
    
    # Robot specifications
    REACH = 0.8  # meters
    PAYLOAD = 7.0  # kg
    DOF = 7
    
    def __init__(self, base_position: List[float] = [0, 0, 0], 
                 urdf_path: str = "data/robot_models/kuka_iiwa/model.urdf"):
        """
        Initialize KUKA iiwa robot.
        
        Args:
            base_position: Robot base position [x, y, z]
            urdf_path: Path to KUKA iiwa URDF file
        """
        # Set KUKA-specific parameters
        self.joint_limits = self.JOINT_LIMITS
        self.home_position = self.HOME_POSITION
        self.end_effector_index = 6  # KUKA iiwa end-effector link
        
        # Use simplified URDF if the specific one doesn't exist
        if not os.path.exists(urdf_path):
            logger.warning(f"KUKA URDF not found at {urdf_path}, using fallback")
            urdf_path = self._create_simple_kuka_urdf()
        
        super().__init__(urdf_path, base_position)
        
        logger.info("KUKA iiwa robot initialized")
    
    def _create_simple_kuka_urdf(self) -> str:
        """
        Create a simplified KUKA iiwa URDF for testing.
        
        Returns:
            Path to created URDF file
        """
        try:
            # Create directory if it doesn't exist
            urdf_dir = "data/robot_models/kuka_iiwa"
            os.makedirs(urdf_dir, exist_ok=True)
            
            urdf_path = os.path.join(urdf_dir, "simple_kuka.urdf")
            
            # Simple KUKA iiwa URDF content
            urdf_content = '''<?xml version="1.0"?>
<robot name="kuka_iiwa">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.15" radius="0.1"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.15" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="link_1">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.06"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.97" upper="2.97" effort="150" velocity="1.7"/>
  </joint>

  <link name="link_2">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.06"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="joint_2" type="revolute">
    <parent link="link_1"/>
    <child link="link_2"/>
    <origin xyz="0 0 0.2" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.09" upper="2.09" effort="150" velocity="1.7"/>
  </joint>

  <link name="link_3">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.06"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="joint_3" type="revolute">
    <parent link="link_2"/>
    <child link="link_3"/>
    <origin xyz="0 0 0.2" rpy="-1.57 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.97" upper="2.97" effort="150" velocity="1.7"/>
  </joint>

  <link name="link_4">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.06"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="joint_4" type="revolute">
    <parent link="link_3"/>
    <child link="link_4"/>
    <origin xyz="0 0 0.2" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.09" upper="2.09" effort="150" velocity="1.7"/>
  </joint>

  <link name="link_5">
    <visual>
      <geometry>
        <cylinder length="0.15" radius="0.05"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.15" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="joint_5" type="revolute">
    <parent link="link_4"/>
    <child link="link_5"/>
    <origin xyz="0 0 0.2" rpy="-1.57 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.97" upper="2.97" effort="150" velocity="1.7"/>
  </joint>

  <link name="link_6">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.04"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="joint_6" type="revolute">
    <parent link="link_5"/>
    <child link="link_6"/>
    <origin xyz="0 0 0.15" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.09" upper="2.09" effort="150" velocity="1.7"/>
  </joint>

  <link name="link_7">
    <visual>
      <geometry>
        <cylinder length="0.08" radius="0.03"/>
      </geometry>
      <material name="grey">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.08" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="joint_7" type="revolute">
    <parent link="link_6"/>
    <child link="link_7"/>
    <origin xyz="0 0 0.1" rpy="-1.57 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.05" upper="3.05" effort="150" velocity="1.7"/>
  </joint>

  <!-- Simple gripper -->
  <link name="gripper_base">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.02"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="gripper_joint" type="fixed">
    <parent link="link_7"/>
    <child link="gripper_base"/>
    <origin xyz="0 0 0.08" rpy="0 0 0"/>
  </joint>

  <link name="gripper_finger_1">
    <visual>
      <geometry>
        <box size="0.02 0.01 0.04"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.01 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="gripper_finger_1_joint" type="prismatic">
    <parent link="gripper_base"/>
    <child link="gripper_finger_1"/>
    <origin xyz="0.02 0 0.01" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.04" effort="10" velocity="0.1"/>
  </joint>

  <link name="gripper_finger_2">
    <visual>
      <geometry>
        <box size="0.02 0.01 0.04"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.01 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="gripper_finger_2_joint" type="prismatic">
    <parent link="gripper_base"/>
    <child link="gripper_finger_2"/>
    <origin xyz="-0.02 0 0.01" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.04" upper="0" effort="10" velocity="0.1"/>
  </joint>
</robot>'''
            
            # Write URDF file
            with open(urdf_path, 'w') as f:
                f.write(urdf_content)
            
            logger.info(f"Created simple KUKA URDF at {urdf_path}")
            return urdf_path
            
        except Exception as e:
            logger.error(f"Failed to create simple KUKA URDF: {e}")
            # Fallback to a basic URDF from pybullet_data
            return "kuka_iiwa/model.urdf"
    
    def get_workspace_limits(self) -> dict:
        """
        Get workspace limits for KUKA iiwa.
        
        Returns:
            Dictionary with workspace boundaries
        """
        return {
            'x_min': -self.REACH,
            'x_max': self.REACH,
            'y_min': -self.REACH,
            'y_max': self.REACH,
            'z_min': 0.1,  # Above table
            'z_max': self.REACH + 0.5  # Maximum reach height
        }
    
    def is_within_workspace(self, position: List[float]) -> bool:
        """
        Check if position is within KUKA iiwa workspace.
        
        Args:
            position: Target position [x, y, z]
            
        Returns:
            True if position is reachable
        """
        limits = self.get_workspace_limits()
        x, y, z = position
        
        # Check Cartesian limits
        if not (limits['x_min'] <= x <= limits['x_max']):
            return False
        if not (limits['y_min'] <= y <= limits['y_max']):
            return False
        if not (limits['z_min'] <= z <= limits['z_max']):
            return False
        
        # Check radial distance from base
        radial_distance = (x**2 + y**2 + z**2)**0.5
        if radial_distance > self.REACH:
            return False
        
        return True
    
    def get_safe_positions(self) -> dict:
        """
        Get predefined safe positions for KUKA iiwa.
        
        Returns:
            Dictionary of safe joint configurations
        """
        return {
            'home': self.HOME_POSITION,
            'vertical': [0, 0, 0, -1.57, 0, 0, 0],
            'folded': [0, 1.57, 0, -1.57, 0, 0, 0],
            'extended': [0, 0, 0, 0, 0, 1.57, 0],
            'inspection': [1.57, 0, 0, -1.57, 0, 1.57, 0]
        }
    
    def move_to_safe_position(self, position_name: str) -> bool:
        """
        Move to a predefined safe position.
        
        Args:
            position_name: Name of safe position
            
        Returns:
            True if movement successful
        """
        safe_positions = self.get_safe_positions()
        
        if position_name not in safe_positions:
            logger.error(f"Unknown safe position: {position_name}")
            return False
        
        target_joints = safe_positions[position_name]
        self.set_joint_positions(target_joints)
        
        logger.info(f"Moved to safe position: {position_name}")
        return True
    
    def get_robot_specifications(self) -> dict:
        """
        Get KUKA iiwa specifications.
        
        Returns:
            Robot specifications dictionary
        """
        return {
            'name': 'KUKA iiwa',
            'dof': self.DOF,
            'reach': self.REACH,
            'payload': self.PAYLOAD,
            'joint_limits': self.JOINT_LIMITS,
            'home_position': self.HOME_POSITION,
            'workspace': self.get_workspace_limits(),
            'safe_positions': list(self.get_safe_positions().keys())
        }
    
    def validate_joint_configuration(self, joint_positions: List[float]) -> bool:
        """
        Validate if joint configuration is within limits.
        
        Args:
            joint_positions: Joint angles in radians
            
        Returns:
            True if configuration is valid
        """
        if len(joint_positions) != self.DOF:
            return False
        
        for i, (pos, (lower, upper)) in enumerate(zip(joint_positions, self.JOINT_LIMITS)):
            if not (lower <= pos <= upper):
                logger.warning(f"Joint {i+1} position {pos:.3f} outside limits [{lower:.3f}, {upper:.3f}]")
                return False
        
        return True
    
    def get_joint_velocities(self) -> List[float]:
        """
        Get current joint velocities.
        
        Returns:
            List of joint velocities
        """
        if self.robot_id is None:
            return []
        
        import pybullet as p
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        velocities = [state[1] for state in joint_states]
        
        return velocities
    
    def emergency_stop(self) -> None:
        """Emergency stop - immediately halt all motion."""
        if self.robot_id is None:
            return
        
        import pybullet as p
        
        # Set all joint velocities to zero
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.VELOCITY_CONTROL,
            targetVelocities=[0] * len(self.joint_indices),
            forces=[1000] * len(self.joint_indices)  # High force to stop quickly
        )
        
        logger.warning("EMERGENCY STOP ACTIVATED")
    
    def self_collision_check(self) -> bool:
        """
        Check for self-collision.
        
        Returns:
            True if robot is in collision with itself
        """
        if self.robot_id is None:
            return False
        
        import pybullet as p
        
        # Get collision information
        contact_points = p.getContactPoints(self.robot_id, self.robot_id)
        
        # Filter out adjacent link contacts (normal)
        actual_collisions = []
        for contact in contact_points:
            link1 = contact[3]
            link2 = contact[4]
            # Ignore contacts between adjacent links
            if abs(link1 - link2) > 1:
                actual_collisions.append(contact)
        
        if actual_collisions:
            logger.warning(f"Self-collision detected: {len(actual_collisions)} contact points")
            return True
        
        return False 
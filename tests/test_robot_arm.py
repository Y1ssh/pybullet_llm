"""
Unit tests for robot arm functionality.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.robot_arm import RobotArm
from src.core.environment import RobotEnvironment
from src.robots.kuka_iiwa import KukaIiwa

class TestRobotArm(unittest.TestCase):
    """Test cases for RobotArm class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = RobotEnvironment(gui_mode=False)  # No GUI for testing
        self.env.setup_physics()
        
        self.robot = KukaIiwa()
        self.robot.load_robot(self.env.physics_client)
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.cleanup()
    
    def test_robot_initialization(self):
        """Test robot initialization."""
        self.assertIsNotNone(self.robot.robot_id)
        self.assertTrue(len(self.robot.joint_indices) > 0)
        self.assertEqual(len(self.robot.home_position), 7)  # KUKA iiwa has 7 joints
    
    def test_joint_positions(self):
        """Test getting and setting joint positions."""
        # Get initial positions
        initial_positions = self.robot.get_joint_positions()
        self.assertEqual(len(initial_positions), 7)
        
        # Set new positions
        test_positions = [0.1, 0.2, 0.3, -1.0, 0.5, 1.0, 0.7]
        self.robot.set_joint_positions(test_positions)
        
        # Allow simulation to settle
        self.env.step_simulation(steps=100)
        
        # Check if positions were set (with some tolerance)
        current_positions = self.robot.get_joint_positions()
        for expected, actual in zip(test_positions, current_positions):
            self.assertAlmostEqual(expected, actual, places=2)
    
    def test_end_effector_position(self):
        """Test end-effector position retrieval."""
        position = self.robot.get_end_effector_position()
        self.assertEqual(len(position), 3)  # x, y, z coordinates
        self.assertIsInstance(position[0], float)
        self.assertIsInstance(position[1], float)
        self.assertIsInstance(position[2], float)
    
    def test_home_position_reset(self):
        """Test reset to home position."""
        # Move to a different position first
        test_positions = [0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5]
        self.robot.set_joint_positions(test_positions)
        self.env.step_simulation(steps=100)
        
        # Reset to home
        self.robot.reset_to_home()
        self.env.step_simulation(steps=100)
        
        # Check if close to home position
        current_positions = self.robot.get_joint_positions()
        for expected, actual in zip(self.robot.home_position, current_positions):
            self.assertAlmostEqual(expected, actual, places=1)
    
    def test_gripper_control(self):
        """Test gripper open/close functionality."""
        # Test opening gripper
        self.robot.open_gripper()
        self.env.step_simulation(steps=50)
        self.assertTrue(self.robot.get_gripper_state())
        
        # Test closing gripper
        self.robot.close_gripper()
        self.env.step_simulation(steps=50)
        self.assertFalse(self.robot.get_gripper_state())
    
    def test_inverse_kinematics(self):
        """Test inverse kinematics calculation."""
        target_position = [0.3, 0.3, 0.8]
        
        # Calculate IK
        joint_positions = self.robot.inverse_kinematics(target_position)
        
        # Check if valid solution
        self.assertIsNotNone(joint_positions)
        self.assertEqual(len(joint_positions), 7)
        
        # Verify by forward kinematics
        calculated_position = self.robot.forward_kinematics(joint_positions)
        
        # Check if close to target (within 5cm tolerance)
        error = np.linalg.norm(np.array(target_position) - np.array(calculated_position))
        self.assertLess(error, 0.05)
    
    def test_workspace_limits(self):
        """Test workspace limit checking (KUKA-specific)."""
        if isinstance(self.robot, KukaIiwa):
            # Test position within workspace
            valid_position = [0.3, 0.3, 0.8]
            self.assertTrue(self.robot.is_within_workspace(valid_position))
            
            # Test position outside workspace
            invalid_position = [2.0, 2.0, 2.0]  # Too far
            self.assertFalse(self.robot.is_within_workspace(invalid_position))
    
    def test_joint_limits_validation(self):
        """Test joint limits validation (KUKA-specific)."""
        if isinstance(self.robot, KukaIiwa):
            # Test valid joint configuration
            valid_joints = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]
            self.assertTrue(self.robot.validate_joint_configuration(valid_joints))
            
            # Test invalid joint configuration (exceeds limits)
            invalid_joints = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
            self.assertFalse(self.robot.validate_joint_configuration(invalid_joints))
    
    def test_robot_specifications(self):
        """Test robot specifications retrieval (KUKA-specific)."""
        if isinstance(self.robot, KukaIiwa):
            specs = self.robot.get_robot_specifications()
            
            self.assertIn('name', specs)
            self.assertIn('dof', specs)
            self.assertIn('reach', specs)
            self.assertIn('payload', specs)
            self.assertEqual(specs['dof'], 7)
            self.assertEqual(specs['name'], 'KUKA iiwa')

class TestRobotEnvironment(unittest.TestCase):
    """Test cases for RobotEnvironment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = RobotEnvironment(gui_mode=False)
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.cleanup()
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.env.setup_physics()
        self.assertIsNotNone(self.env.physics_client)
        self.assertIsNotNone(self.env.plane_id)
    
    def test_object_spawning(self):
        """Test object spawning functionality."""
        self.env.setup_physics()
        
        # Spawn a test object
        object_id = self.env.spawn_object(
            "box", 
            [0.0, 0.0, 1.0], 
            [1, 0, 0, 1], 
            [0.1, 0.1, 0.1]
        )
        
        self.assertIsNotNone(object_id)
        self.assertIn(object_id, self.env.objects)
        
        # Test object removal
        self.env.remove_object(object_id)
        self.assertNotIn(object_id, self.env.objects)
    
    def test_camera_setup(self):
        """Test camera configuration."""
        self.env.setup_physics()
        self.env.setup_camera()
        
        self.assertTrue(self.env.camera_params)
        self.assertIn('width', self.env.camera_params)
        self.assertIn('height', self.env.camera_params)
    
    def test_simulation_stepping(self):
        """Test simulation stepping."""
        self.env.setup_physics()
        
        # Should not raise any exceptions
        self.env.step_simulation(steps=10)

if __name__ == '__main__':
    unittest.main() 
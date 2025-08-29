#!/usr/bin/env python3
"""
Basic demonstration of PyBullet LLM robotics functionality.
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.environment import RobotEnvironment
from src.robots.kuka_iiwa import KukaIiwa
from src.core.sensors import CameraSystem
from src.llm.llm_controller import LLMRobotController

def main():
    """Run basic demonstration."""
    print("🚀 Starting PyBullet LLM Robotics Demo")
    
    # Setup environment
    print("📦 Setting up environment...")
    env = RobotEnvironment(gui_mode=True)
    env.setup_physics()
    
    # Initialize robot
    print("🦾 Loading KUKA iiwa robot...")
    robot = KukaIiwa()
    robot.load_robot(env.physics_client)
    
    # Setup camera
    print("📷 Configuring camera...")
    camera = CameraSystem(env)
    env.setup_camera()
    
    # Add demo objects
    print("📦 Adding objects to scene...")
    env.spawn_object("box", [0.3, 0.2, 0.78], [1, 0, 0, 1], [0.03, 0.03, 0.03])  # Red cube
    env.spawn_object("sphere", [0.3, -0.2, 0.78], [0, 0, 1, 1], [0.03, 0.03, 0.03])  # Blue sphere
    
    print("✅ Setup complete!")
    print("\n" + "="*60)
    
    # Demonstrate basic robot control
    print("🎯 Demonstrating basic robot control...")
    
    # Move to a position
    print("Moving robot to position [0.4, 0.3, 0.9]...")
    success = robot.move_to_position([0.4, 0.3, 0.9])
    if success:
        print("✅ Movement successful!")
    else:
        print("❌ Movement failed!")
    
    env.step_simulation(steps=100)
    time.sleep(1)
    
    # Open and close gripper
    print("Opening gripper...")
    robot.open_gripper()
    env.step_simulation(steps=50)
    time.sleep(0.5)
    
    print("Closing gripper...")
    robot.close_gripper()
    env.step_simulation(steps=50)
    time.sleep(0.5)
    
    # Reset to home
    print("Resetting to home position...")
    robot.reset_to_home()
    env.step_simulation(steps=100)
    time.sleep(1)
    
    # Demonstrate computer vision
    print("\n🔍 Demonstrating computer vision...")
    rgb_image = camera.capture_rgb_image()
    detected_objects = camera.detect_objects_in_view(rgb_image)
    
    print(f"Detected {len(detected_objects)} objects:")
    for i, obj in enumerate(detected_objects):
        print(f"  {i+1}. {obj['color']} {obj['shape']} at {obj['center']}")
    
    # Try LLM control (if API keys are available)
    print("\n🧠 Testing LLM integration...")
    try:
        controller = LLMRobotController(robot, env, camera, llm_provider="anthropic")
        print("LLM controller initialized successfully!")
        
        # Test a simple command
        response = controller.chat_with_robot("scan the environment")
        print(f"LLM Response: {response}")
        
    except Exception as e:
        print(f"LLM integration not available: {e}")
        print("Make sure to set your API keys in the .env file")
    
    print("\n" + "="*60)
    print("🎉 Demo complete! Press Enter to exit...")
    input()
    
    # Cleanup
    print("🧹 Cleaning up...")
    env.cleanup()
    print("✅ Goodbye!")

if __name__ == "__main__":
    main() 
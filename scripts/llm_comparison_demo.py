#!/usr/bin/env python3
"""
Demonstration script for comparing different LLM providers.
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.environment import RobotEnvironment
from src.robots.kuka_iiwa import KukaIiwa
from src.core.sensors import CameraSystem
from src.llm.llm_controller import LLMRobotController

def compare_llm_providers(controller, test_commands):
    """
    Compare responses from different LLM providers.
    
    Args:
        controller: LLMRobotController instance
        test_commands: List of commands to test
    """
    print("🧠 LLM Provider Comparison Demo")
    print("="*60)
    
    available_providers = controller.get_available_providers()
    print(f"Available providers: {', '.join(available_providers)}")
    
    if len(available_providers) < 2:
        print("⚠️  Need at least 2 LLM providers for comparison")
        print("Make sure to configure API keys for multiple providers in .env")
        return
    
    for i, command in enumerate(test_commands, 1):
        print(f"\n🎯 Test Command {i}: '{command}'")
        print("-" * 50)
        
        # Get responses from all providers
        comparison = controller.compare_llm_responses(command)
        
        for provider, response in comparison.items():
            print(f"\n{provider.upper()}:")
            print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        print("\n" + "="*60)
        
        # Wait for user input to continue
        if i < len(test_commands):
            input("Press Enter to continue to next test...")

def interactive_comparison(controller):
    """
    Interactive mode for LLM comparison.
    
    Args:
        controller: LLMRobotController instance
    """
    print("\n🎮 Interactive LLM Comparison Mode")
    print("Enter commands to compare responses from different providers")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            command = input("\n🎤 Enter command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            if not command:
                continue
            
            print(f"\n🔄 Comparing responses for: '{command}'")
            comparison = controller.compare_llm_responses(command)
            
            print("\n📊 Comparison Results:")
            print("="*40)
            
            for provider, response in comparison.items():
                print(f"\n{provider.upper()}:")
                print(f"{response}")
                print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="LLM Provider Comparison Demo")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--no-gui", action="store_true", 
                       help="Run without PyBullet GUI")
    
    args = parser.parse_args()
    
    print("🚀 Starting LLM Provider Comparison Demo")
    
    try:
        # Setup minimal environment for LLM testing
        print("📦 Setting up environment...")
        env = RobotEnvironment(gui_mode=not args.no_gui)
        env.setup_physics()
        
        # Initialize robot
        print("🦾 Loading robot...")
        robot = KukaIiwa()
        robot.load_robot(env.physics_client)
        
        # Setup camera
        print("📷 Configuring camera...")
        camera = CameraSystem(env)
        env.setup_camera()
        
        # Add some objects for context
        env.spawn_object("box", [0.3, 0.2, 0.78], [1, 0, 0, 1], [0.03, 0.03, 0.03])
        env.spawn_object("sphere", [0.3, -0.2, 0.78], [0, 0, 1, 1], [0.03, 0.03, 0.03])
        
        # Initialize LLM controller
        print("🧠 Initializing LLM controller...")
        controller = LLMRobotController(robot, env, camera)
        
        print("✅ Setup complete!\n")
        
        if args.interactive:
            interactive_comparison(controller)
        else:
            # Predefined test commands
            test_commands = [
                "How would you pick up the red cube safely?",
                "Explain the best strategy for stacking objects",
                "What should I do if the robot arm gets stuck?",
                "How do you handle delicate objects?",
                "Describe the steps to reset the robot to home position"
            ]
            
            compare_llm_providers(controller, test_commands)
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        env.cleanup()
        print("✅ Demo complete!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure your API keys are configured in .env file")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
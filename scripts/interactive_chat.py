#!/usr/bin/env python3
"""
Interactive chat interface for PyBullet LLM robot control.
"""

import sys
import os
import logging
import argparse
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.robot_arm import RobotArm
from src.core.environment import RobotEnvironment
from src.core.sensors import CameraSystem
from src.robots.kuka_iiwa import KukaIiwa
from src.llm.llm_controller import LLMRobotController
from src.utils.config import llm_config, robot_config, simulation_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveRobotChat:
    """Interactive chat interface for robot control."""
    
    def __init__(self, gui_mode: bool = True, llm_provider: str = "anthropic"):
        """
        Initialize interactive robot chat.
        
        Args:
            gui_mode: Whether to use PyBullet GUI
            llm_provider: LLM provider to use
        """
        self.gui_mode = gui_mode
        self.llm_provider = llm_provider
        self.environment = None
        self.robot_arm = None
        self.camera_system = None
        self.controller = None
        self.running = False
        
    def setup_system(self) -> bool:
        """
        Setup the complete robot system.
        
        Returns:
            True if setup successful
        """
        try:
            print("🤖 Initializing PyBullet LLM Robot System...")
            
            # Setup environment
            print("📦 Setting up environment...")
            self.environment = RobotEnvironment(
                gui_mode=self.gui_mode, 
                timestep=simulation_config.timestep
            )
            self.environment.setup_physics()
            
            # Initialize robot
            print("🦾 Loading robot...")
            self.robot_arm = KukaIiwa(base_position=robot_config.base_position)
            robot_id = self.environment.load_robot(
                self.robot_arm.urdf_path, 
                self.robot_arm.base_position
            )
            self.robot_arm.load_robot(self.environment.physics_client)
            
            # Setup camera
            print("📷 Configuring camera...")
            self.camera_system = CameraSystem(self.environment)
            self.environment.setup_camera()
            
            # Add some objects to the scene
            print("📦 Adding objects to scene...")
            self._add_demo_objects()
            
            # Initialize LLM controller
            print(f"🧠 Initializing LLM controller with {self.llm_provider}...")
            self.controller = LLMRobotController(
                robot_arm=self.robot_arm,
                environment=self.environment,
                camera_system=self.camera_system,
                llm_provider=self.llm_provider
            )
            
            print("✅ System initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup system: {e}")
            print(f"❌ Setup failed: {e}")
            return False
    
    def _add_demo_objects(self) -> None:
        """Add demonstration objects to the scene."""
        try:
            # Add some colored objects on the table
            table_height = 0.78  # Slightly above table surface
            
            # Red cube
            self.environment.spawn_object(
                "box", 
                [0.3, 0.2, table_height], 
                [1, 0, 0, 1],  # Red
                [0.03, 0.03, 0.03]
            )
            
            # Blue sphere
            self.environment.spawn_object(
                "sphere", 
                [0.3, -0.2, table_height], 
                [0, 0, 1, 1],  # Blue
                [0.03, 0.03, 0.03]
            )
            
            # Green cylinder
            self.environment.spawn_object(
                "cylinder", 
                [-0.3, 0.1, table_height], 
                [0, 1, 0, 1],  # Green
                [0.03, 0.03, 0.06]
            )
            
            # Yellow box
            self.environment.spawn_object(
                "box", 
                [-0.2, -0.3, table_height], 
                [1, 1, 0, 1],  # Yellow
                [0.04, 0.02, 0.03]
            )
            
            print("Added demo objects: red cube, blue sphere, green cylinder, yellow box")
            
        except Exception as e:
            logger.warning(f"Failed to add demo objects: {e}")
    
    def print_welcome_message(self) -> None:
        """Print welcome message and instructions."""
        welcome_msg = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🤖 PyBullet LLM Robot Controller 🤖                       ║
║                                                                              ║
║  Welcome to the interactive robot control system!                           ║
║  You can control the robot using natural language commands.                 ║
║                                                                              ║
║  Example commands:                                                           ║
║  • "scan the environment"                                                    ║
║  • "pick up the red cube"                                                    ║
║  • "move to position 0.3, 0.4, 0.5"                                        ║
║  • "place the object at 0.2, 0.3, 0.8"                                     ║
║  • "switch to OpenAI"                                                        ║
║  • "compare LLM responses for 'pick up the blue sphere'"                    ║
║                                                                              ║
║  Special commands:                                                           ║
║  • "help" - Show detailed help                                              ║
║  • "status" - Show system status                                            ║
║  • "providers" - List available LLM providers                               ║
║  • "quit" or "exit" - Exit the program                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(welcome_msg)
        
        # Show current LLM provider
        if self.controller:
            current_provider = self.controller.llm_manager.current_provider
            available_providers = self.controller.get_available_providers()
            print(f"🧠 Current LLM Provider: {current_provider}")
            print(f"📋 Available Providers: {', '.join(available_providers)}")
        
        print("\n" + "="*80)
    
    def handle_special_commands(self, user_input: str) -> bool:
        """
        Handle special system commands.
        
        Args:
            user_input: User input string
            
        Returns:
            True if command was handled (don't pass to LLM)
        """
        command = user_input.lower().strip()
        
        if command in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            self.running = False
            return True
        
        elif command == 'help':
            if self.controller:
                help_text = self.controller.get_help_text()
                print(help_text)
            else:
                print("❌ Controller not initialized")
            return True
        
        elif command == 'status':
            self._show_system_status()
            return True
        
        elif command == 'providers':
            self._show_llm_providers()
            return True
        
        elif command == 'clear':
            if self.controller:
                self.controller.clear_conversation_history()
                print("🧹 Conversation history cleared")
            return True
        
        elif command.startswith('switch to '):
            provider = command.replace('switch to ', '').strip()
            if self.controller:
                success = self.controller.switch_llm_provider(provider)
                if success:
                    print(f"✅ Switched to {provider}")
                else:
                    print(f"❌ Failed to switch to {provider}")
            return True
        
        return False
    
    def _show_system_status(self) -> None:
        """Show comprehensive system status."""
        if not self.controller:
            print("❌ Controller not initialized")
            return
        
        try:
            status = self.controller.get_system_status()
            
            print("\n📊 System Status:")
            print("="*50)
            
            # Robot status
            if 'robot' in status and status['robot']:
                robot_info = status['robot']
                print(f"🦾 Robot: Online")
                print(f"   End-effector: {robot_info.get('end_effector_position', 'Unknown')}")
                print(f"   Gripper: {'Open' if robot_info.get('gripper_open', True) else 'Closed'}")
                print(f"   Joints: {robot_info.get('controllable_joints', 0)}")
            else:
                print("🦾 Robot: Offline")
            
            # Environment status
            if 'environment' in status and status['environment']:
                env_info = status['environment']
                print(f"🌍 Environment: Online")
                print(f"   Objects: {env_info.get('num_objects', 0)}")
                print(f"   Camera: {'Configured' if env_info.get('camera_configured', False) else 'Not configured'}")
            else:
                print("🌍 Environment: Offline")
            
            # LLM status
            if 'llm' in status and status['llm']:
                llm_info = status['llm']
                print(f"🧠 LLM: {llm_info.get('current_provider', 'Unknown')}")
                print(f"   Available: {', '.join(llm_info.get('available_providers', []))}")
                print(f"   Total providers: {llm_info.get('total_providers', 0)}")
            
            # Conversation
            conv_length = status.get('conversation_length', 0)
            print(f"💬 Conversation: {conv_length} messages")
            
            print("="*50)
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    def _show_llm_providers(self) -> None:
        """Show available LLM providers and their status."""
        if not self.controller:
            print("❌ Controller not initialized")
            return
        
        try:
            available = self.controller.get_available_providers()
            current = self.controller.llm_manager.current_provider
            
            print("\n🧠 LLM Providers:")
            print("="*40)
            
            for provider in available:
                status_icon = "✅" if provider == current else "⚪"
                capabilities = self.controller.llm_manager.get_provider_capabilities(provider)
                model = capabilities.get('model', 'Unknown')
                print(f"{status_icon} {provider}: {model}")
            
            print("="*40)
            print(f"Current: {current}")
            print("\nTo switch providers, use: 'switch to <provider_name>'")
            
        except Exception as e:
            print(f"❌ Error getting provider info: {e}")
    
    def run_chat_loop(self) -> None:
        """Run the main chat loop."""
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n🎤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if self.handle_special_commands(user_input):
                    continue
                
                # Process with LLM controller
                if self.controller:
                    print("🤖 Robot: ", end="", flush=True)
                    response = self.controller.chat_with_robot(user_input)
                    print(response)
                else:
                    print("❌ Controller not available")
                
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                self.running = False
            except EOFError:
                print("\n\n👋 Input ended")
                self.running = False
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"❌ Error: {e}")
    
    def cleanup(self) -> None:
        """Cleanup system resources."""
        try:
            print("\n🧹 Cleaning up...")
            
            if self.controller:
                self.controller.cleanup()
            
            if self.environment:
                self.environment.cleanup()
            
            print("✅ Cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            print(f"❌ Cleanup error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="PyBullet LLM Robot Interactive Chat")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--provider", default="anthropic", 
                       choices=["anthropic", "openai", "google"],
                       help="LLM provider to use")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run chat interface
    chat = InteractiveRobotChat(
        gui_mode=not args.no_gui,
        llm_provider=args.provider
    )
    
    try:
        # Setup system
        if not chat.setup_system():
            print("❌ Failed to setup system. Exiting.")
            return 1
        
        # Show welcome message
        chat.print_welcome_message()
        
        # Run chat loop
        chat.run_chat_loop()
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        return 1
    
    finally:
        # Cleanup
        chat.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main()) 
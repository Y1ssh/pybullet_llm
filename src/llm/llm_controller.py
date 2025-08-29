"""
Main LLM controller for natural language robot control.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.robot_arm import RobotArm
from ..core.environment import RobotEnvironment
from ..core.sensors import CameraSystem
from ..robots.llm_provider_manager import LLMProviderManager
from .robot_tools import ROBOT_TOOLS, set_robot_components

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMRobotController:
    """Main controller for LLM-based robot control."""
    
    def __init__(self, robot_arm: RobotArm, environment: RobotEnvironment, 
                 camera_system: CameraSystem, llm_model: str = "claude-3-5-sonnet-20241022", 
                 llm_provider: str = "anthropic"):
        """
        Initialize LLM robot controller.
        
        Args:
            robot_arm: Robot arm instance
            environment: Robot environment instance
            camera_system: Camera system instance
            llm_model: LLM model name
            llm_provider: LLM provider name
        """
        self.robot_arm = robot_arm
        self.environment = environment
        self.camera_system = camera_system
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        
        # Initialize LLM provider manager
        self.llm_manager = LLMProviderManager()
        
        # Set global components for tools
        set_robot_components(robot_arm, environment, camera_system, self.llm_manager)
        
        # Initialize agent
        self.agent = None
        self.agent_executor = None
        self._setup_agent()
        
        # Conversation history
        self.conversation_history = []
        
        logger.info(f"LLM Robot Controller initialized with {llm_provider} provider")
    
    def _setup_agent(self) -> None:
        """Setup the LangChain agent with tools."""
        try:
            # Get current LLM provider
            llm = self.llm_manager.get_provider()
            
            # Create system prompt
            system_prompt = self.get_system_prompt()
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            # Create agent
            self.agent = create_tool_calling_agent(llm, ROBOT_TOOLS, prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=ROBOT_TOOLS,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10
            )
            
            logger.info("Agent successfully created and configured")
            
        except Exception as e:
            logger.error(f"Failed to setup agent: {e}")
            raise
    
    def get_system_prompt(self) -> str:
        """
        Get system prompt for the robot control agent.
        
        Returns:
            System prompt string
        """
        prompt = """You are an intelligent robot control assistant. You can control a robotic arm in a PyBullet simulation environment through natural language commands.

Available capabilities:
- Move the robot arm to specific positions
- Pick up objects using visual detection
- Place objects at target locations
- Open and close the gripper
- Scan the environment to see what objects are available
- Get robot status and positions
- Reset robot to home position
- Switch between different LLM providers
- Compare responses from multiple LLM providers

Important guidelines:
1. Always be precise when moving the robot - check positions and validate movements
2. Use the scan_environment tool to understand what objects are available before attempting to pick them up
3. When picking up objects, be specific about color and shape descriptions
4. Always ensure the gripper is in the correct state (open/closed) for the task
5. Use get_robot_status to check current robot state when needed
6. Be safe - avoid movements that could cause collisions
7. If a command fails, try to diagnose the issue and suggest alternatives
8. When placing objects, ensure they are placed on stable surfaces (like the table)
9. Provide clear feedback about what actions you're taking and their results

The robot workspace typically includes:
- A table at approximately (0, 0, 0.75) height
- Objects may be placed on the table surface
- Robot base is at (0, 0, 0)
- Typical reachable positions are within 0.5-1.0 meters from the base

You can switch between different LLM providers (anthropic, openai, google) and compare their responses using the provided tools. This allows you to leverage different AI models for different types of tasks.

Be helpful, precise, and always prioritize safety in robot operations."""

        return prompt
    
    def chat_with_robot(self, user_input: str) -> str:
        """
        Process natural language input and execute robot commands.
        
        Args:
            user_input: Natural language command from user
            
        Returns:
            Response from the agent
        """
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Process command with agent
            result = self.agent_executor.invoke({
                "input": user_input
            })
            
            response = result.get("output", "No response generated")
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            logger.info(f"Processed command: {user_input}")
            return response
            
        except Exception as e:
            error_msg = f"Error processing command: {e}"
            logger.error(error_msg)
            return error_msg
    
    def process_natural_language_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command and return structured result.
        
        Args:
            command: Natural language command
            
        Returns:
            Dictionary with command processing results
        """
        try:
            result = self.agent_executor.invoke({"input": command})
            
            return {
                "success": True,
                "command": command,
                "response": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "provider": self.llm_manager.current_provider
            }
            
        except Exception as e:
            return {
                "success": False,
                "command": command,
                "error": str(e),
                "provider": self.llm_manager.current_provider
            }
    
    def handle_error_recovery(self, error: Exception) -> str:
        """
        Handle error recovery and provide helpful feedback.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Recovery suggestion string
        """
        error_msg = str(error).lower()
        
        if "not reachable" in error_msg:
            return "The target position is outside the robot's workspace. Try a position closer to the robot base (within 1 meter)."
        elif "object not found" in error_msg:
            return "The specified object was not detected. Use 'scan environment' to see available objects, or try describing the object differently."
        elif "gripper" in error_msg:
            return "There was an issue with the gripper. Check if it's in the correct state (open/closed) for the task."
        elif "robot not initialized" in error_msg:
            return "The robot system is not properly initialized. Please restart the system."
        elif "camera" in error_msg:
            return "There was an issue with the camera system. Check if the camera is properly configured."
        else:
            return f"An unexpected error occurred: {error}. Please try again or rephrase your command."
    
    def switch_llm_provider(self, provider: str) -> bool:
        """
        Switch to a different LLM provider.
        
        Args:
            provider: Provider name (anthropic, openai, google)
            
        Returns:
            True if switch successful
        """
        try:
            success = self.llm_manager.switch_provider(provider)
            if success:
                # Recreate agent with new provider
                self._setup_agent()
                logger.info(f"Switched to {provider} provider and recreated agent")
            return success
        except Exception as e:
            logger.error(f"Failed to switch provider: {e}")
            return False
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available LLM providers.
        
        Returns:
            List of available provider names
        """
        return self.llm_manager.get_available_providers()
    
    def compare_llm_responses(self, command: str) -> Dict[str, str]:
        """
        Compare responses from different LLM providers for the same command.
        
        Args:
            command: Command to test with all providers
            
        Returns:
            Dictionary mapping provider names to responses
        """
        return self.llm_manager.compare_providers_response(command)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            System status dictionary
        """
        try:
            # Get robot status
            robot_info = self.robot_arm.get_joint_info() if self.robot_arm else {}
            
            # Get environment status
            env_info = self.environment.get_environment_info() if self.environment else {}
            
            # Get camera status
            camera_info = self.camera_system.get_camera_info() if self.camera_system else {}
            
            # Get LLM status
            llm_status = self.llm_manager.get_status_summary()
            
            status = {
                "robot": robot_info,
                "environment": env_info,
                "camera": camera_info,
                "llm": llm_status,
                "conversation_length": len(self.conversation_history),
                "agent_initialized": self.agent is not None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    def execute_command_safely(self, command: str) -> Dict[str, Any]:
        """
        Execute a command with comprehensive error handling and safety checks.
        
        Args:
            command: Natural language command
            
        Returns:
            Execution result with safety information
        """
        try:
            # Pre-execution safety checks
            safety_checks = self._perform_safety_checks()
            
            if not safety_checks["safe"]:
                return {
                    "success": False,
                    "error": "Safety check failed",
                    "safety_issues": safety_checks["issues"],
                    "command": command
                }
            
            # Execute command
            result = self.process_natural_language_command(command)
            
            # Post-execution validation
            if result["success"]:
                validation = self._validate_execution_result()
                result["validation"] = validation
            
            return result
            
        except Exception as e:
            recovery_suggestion = self.handle_error_recovery(e)
            return {
                "success": False,
                "error": str(e),
                "recovery_suggestion": recovery_suggestion,
                "command": command
            }
    
    def _perform_safety_checks(self) -> Dict[str, Any]:
        """
        Perform safety checks before command execution.
        
        Returns:
            Safety check results
        """
        issues = []
        
        try:
            # Check robot initialization
            if self.robot_arm is None:
                issues.append("Robot arm not initialized")
            
            # Check environment
            if self.environment is None:
                issues.append("Environment not initialized")
            
            # Check if robot is in a safe state
            if self.robot_arm:
                current_pos = self.robot_arm.get_end_effector_position()
                if current_pos[2] < 0.1:  # Too close to ground
                    issues.append("Robot end-effector too close to ground")
            
            return {
                "safe": len(issues) == 0,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "safe": False,
                "issues": [f"Safety check error: {e}"]
            }
    
    def _validate_execution_result(self) -> Dict[str, Any]:
        """
        Validate the result of command execution.
        
        Returns:
            Validation results
        """
        try:
            validation = {
                "robot_responsive": True,
                "position_valid": True,
                "gripper_functional": True
            }
            
            if self.robot_arm:
                # Check if robot is still responsive
                try:
                    current_pos = self.robot_arm.get_end_effector_position()
                    validation["current_position"] = current_pos
                except Exception:
                    validation["robot_responsive"] = False
                
                # Check gripper state
                try:
                    gripper_state = self.robot_arm.get_gripper_state()
                    validation["gripper_state"] = gripper_state
                except Exception:
                    validation["gripper_functional"] = False
            
            return validation
            
        except Exception as e:
            return {"validation_error": str(e)}
    
    def get_help_text(self) -> str:
        """
        Get help text with available commands and examples.
        
        Returns:
            Help text string
        """
        help_text = """
PyBullet LLM Robot Controller - Available Commands:

Basic Movement:
- "Move the robot to position 0.3, 0.4, 0.5"
- "Move to the center of the table"
- "Reset robot to home position"

Object Manipulation:
- "Pick up the red cube"
- "Grab the blue sphere"
- "Place the object at position 0.2, 0.3, 0.8"
- "Put the object on the table"

Gripper Control:
- "Open the gripper"
- "Close the gripper"

Environment Scanning:
- "Scan the environment"
- "What objects can you see?"
- "Show me what's on the table"
- "Find the position of the green block"

Robot Status:
- "What is the robot's current status?"
- "Where is the robot arm now?"
- "Check robot position"

LLM Provider Management:
- "Switch to OpenAI"
- "Change to Google Gemini"
- "Use Anthropic Claude"
- "Compare all LLM responses for 'pick up the red cube'"

Complex Tasks:
- "Stack the blue block on top of the red cube"
- "Move all red objects to the left side of the table"
- "Organize the objects by color"

Tips:
- Be specific about colors and shapes when referring to objects
- Use scan_environment first to see what objects are available
- The robot workspace is typically within 1 meter of the base
- Table surface is at approximately 0.75m height
"""
        return help_text
    
    def cleanup(self) -> None:
        """Cleanup controller resources."""
        try:
            if self.llm_manager:
                self.llm_manager.cleanup()
            
            self.conversation_history.clear()
            self.agent = None
            self.agent_executor = None
            
            logger.info("LLM Robot Controller cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 
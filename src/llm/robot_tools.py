"""
LangChain tools for robot control compatible with all LLM providers.
"""

from langchain.tools import tool
from typing import List, Optional, Dict, Any
import logging
import json

# Global variables to store robot components (will be set by controller)
_robot_arm = None
_environment = None
_camera_system = None
_llm_provider_manager = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_robot_components(robot_arm, environment, camera_system, llm_provider_manager):
    """Set global robot components for tools to use."""
    global _robot_arm, _environment, _camera_system, _llm_provider_manager
    _robot_arm = robot_arm
    _environment = environment
    _camera_system = camera_system
    _llm_provider_manager = llm_provider_manager

@tool
def move_robot_to_position(x: float, y: float, z: float) -> str:
    """Move robot end-effector to specified position (x, y, z)"""
    try:
        if _robot_arm is None:
            return "Error: Robot not initialized"
        
        target_pos = [x, y, z]
        
        # Check if position is reachable
        if not _robot_arm.is_position_reachable(target_pos):
            return f"Error: Position [{x}, {y}, {z}] is not reachable by the robot"
        
        # Move to position
        success = _robot_arm.move_to_position(target_pos)
        
        if success:
            # Step simulation to complete movement
            if _environment:
                _environment.step_simulation(steps=100)
            
            # Get actual final position
            final_pos = _robot_arm.get_end_effector_position()
            return f"Successfully moved robot to position [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]"
        else:
            return f"Failed to move robot to position [{x}, {y}, {z}]"
            
    except Exception as e:
        logger.error(f"Error in move_robot_to_position: {e}")
        return f"Error moving robot: {e}"

@tool
def pick_up_object(object_description: str) -> str:
    """Pick up an object based on description (color, shape, position)"""
    try:
        if _robot_arm is None or _camera_system is None:
            return "Error: Robot or camera system not initialized"
        
        # Capture current scene
        rgb_image = _camera_system.capture_rgb_image()
        detected_objects = _camera_system.detect_objects_in_view(rgb_image)
        
        if not detected_objects:
            return "No objects detected in the scene"
        
        # Find matching object
        target_object = None
        description_lower = object_description.lower()
        
        for obj in detected_objects:
            obj_color = obj.get('color', '').lower()
            obj_shape = obj.get('shape', '').lower()
            
            if (obj_color in description_lower or 
                obj_shape in description_lower or
                description_lower in f"{obj_color} {obj_shape}"):
                target_object = obj
                break
        
        if target_object is None:
            available_objects = [f"{obj['color']} {obj['shape']}" for obj in detected_objects]
            return f"Object '{object_description}' not found. Available objects: {', '.join(available_objects)}"
        
        # Get object world position (simplified)
        center_pixel = target_object['center']
        depth_image = _camera_system.capture_depth_image()
        world_pos = _camera_system.get_object_world_position(
            center_pixel[0], center_pixel[1], depth_image
        )
        
        if world_pos is None:
            return f"Failed to determine 3D position of {object_description}"
        
        # Move above object
        approach_pos = [world_pos[0], world_pos[1], world_pos[2] + 0.1]
        if not _robot_arm.move_to_position(approach_pos):
            return f"Failed to move above {object_description}"
        
        # Open gripper
        _robot_arm.open_gripper()
        _environment.step_simulation(steps=50)
        
        # Move down to object
        grasp_pos = [world_pos[0], world_pos[1], world_pos[2] + 0.02]
        if not _robot_arm.move_to_position(grasp_pos):
            return f"Failed to reach {object_description}"
        
        _environment.step_simulation(steps=50)
        
        # Close gripper
        _robot_arm.close_gripper()
        _environment.step_simulation(steps=50)
        
        # Lift object
        lift_pos = [world_pos[0], world_pos[1], world_pos[2] + 0.15]
        _robot_arm.move_to_position(lift_pos)
        _environment.step_simulation(steps=100)
        
        return f"Successfully picked up {object_description}"
        
    except Exception as e:
        logger.error(f"Error in pick_up_object: {e}")
        return f"Error picking up object: {e}"

@tool
def place_object_at_position(x: float, y: float, z: float) -> str:
    """Place currently held object at specified position"""
    try:
        if _robot_arm is None:
            return "Error: Robot not initialized"
        
        if _robot_arm.get_gripper_state():
            return "No object currently held by gripper"
        
        # Move above target position
        approach_pos = [x, y, z + 0.1]
        if not _robot_arm.move_to_position(approach_pos):
            return f"Failed to move above target position [{x}, {y}, {z}]"
        
        _environment.step_simulation(steps=50)
        
        # Move down to place object
        place_pos = [x, y, z + 0.02]
        if not _robot_arm.move_to_position(place_pos):
            return f"Failed to reach placement position [{x}, {y}, {z}]"
        
        _environment.step_simulation(steps=50)
        
        # Open gripper to release object
        _robot_arm.open_gripper()
        _environment.step_simulation(steps=50)
        
        # Move away
        retreat_pos = [x, y, z + 0.15]
        _robot_arm.move_to_position(retreat_pos)
        _environment.step_simulation(steps=100)
        
        return f"Successfully placed object at position [{x:.3f}, {y:.3f}, {z:.3f}]"
        
    except Exception as e:
        logger.error(f"Error in place_object_at_position: {e}")
        return f"Error placing object: {e}"

@tool
def get_robot_status() -> str:
    """Get current robot status including joint positions and gripper state"""
    try:
        if _robot_arm is None:
            return "Error: Robot not initialized"
        
        joint_info = _robot_arm.get_joint_info()
        end_effector_pos = _robot_arm.get_end_effector_position()
        
        status = {
            "end_effector_position": [round(pos, 3) for pos in end_effector_pos],
            "joint_positions": [round(pos, 3) for pos in joint_info['current_positions']],
            "gripper_open": joint_info['gripper_open'],
            "controllable_joints": joint_info['controllable_joints'],
            "gripper_joints": joint_info['gripper_joints']
        }
        
        return json.dumps(status, indent=2)
        
    except Exception as e:
        logger.error(f"Error in get_robot_status: {e}")
        return f"Error getting robot status: {e}"

@tool
def scan_environment() -> str:
    """Scan environment and describe visible objects with positions"""
    try:
        if _camera_system is None:
            return "Error: Camera system not initialized"
        
        # Capture scene
        rgb_image = _camera_system.capture_rgb_image()
        detected_objects = _camera_system.detect_objects_in_view(rgb_image)
        
        if not detected_objects:
            return "No objects detected in the environment"
        
        # Get object positions
        object_descriptions = []
        for i, obj in enumerate(detected_objects):
            description = f"Object {i+1}: {obj['color']} {obj['shape']}"
            description += f" at pixel position ({obj['center'][0]}, {obj['center'][1]})"
            description += f", size: {obj['size']} pixels"
            object_descriptions.append(description)
        
        # Get environment info
        env_info = _environment.get_environment_info() if _environment else {}
        
        result = f"Environment scan results:\n"
        result += f"Total objects detected: {len(detected_objects)}\n"
        result += f"Objects in scene:\n" + "\n".join(object_descriptions)
        
        if env_info:
            result += f"\n\nEnvironment info:\n"
            result += f"Total objects in simulation: {env_info.get('num_objects', 'Unknown')}\n"
            result += f"Camera configured: {env_info.get('camera_configured', False)}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in scan_environment: {e}")
        return f"Error scanning environment: {e}"

@tool
def reset_robot_position() -> str:
    """Reset robot to home position"""
    try:
        if _robot_arm is None:
            return "Error: Robot not initialized"
        
        _robot_arm.reset_to_home()
        
        if _environment:
            _environment.step_simulation(steps=100)
        
        final_pos = _robot_arm.get_end_effector_position()
        return f"Robot reset to home position. End-effector at [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]"
        
    except Exception as e:
        logger.error(f"Error in reset_robot_position: {e}")
        return f"Error resetting robot: {e}"

@tool
def open_gripper() -> str:
    """Open the robot gripper"""
    try:
        if _robot_arm is None:
            return "Error: Robot not initialized"
        
        _robot_arm.open_gripper()
        
        if _environment:
            _environment.step_simulation(steps=50)
        
        return "Gripper opened successfully"
        
    except Exception as e:
        logger.error(f"Error in open_gripper: {e}")
        return f"Error opening gripper: {e}"

@tool
def close_gripper() -> str:
    """Close the robot gripper"""
    try:
        if _robot_arm is None:
            return "Error: Robot not initialized"
        
        _robot_arm.close_gripper()
        
        if _environment:
            _environment.step_simulation(steps=50)
        
        return "Gripper closed successfully"
        
    except Exception as e:
        logger.error(f"Error in close_gripper: {e}")
        return f"Error closing gripper: {e}"

@tool
def get_object_position(object_description: str) -> str:
    """Get position of specific object by description"""
    try:
        if _camera_system is None:
            return "Error: Camera system not initialized"
        
        # Capture scene
        rgb_image = _camera_system.capture_rgb_image()
        detected_objects = _camera_system.detect_objects_in_view(rgb_image)
        
        if not detected_objects:
            return "No objects detected in the scene"
        
        # Find matching object
        description_lower = object_description.lower()
        matching_objects = []
        
        for obj in detected_objects:
            obj_color = obj.get('color', '').lower()
            obj_shape = obj.get('shape', '').lower()
            
            if (obj_color in description_lower or 
                obj_shape in description_lower or
                description_lower in f"{obj_color} {obj_shape}"):
                
                # Get world position
                center_pixel = obj['center']
                depth_image = _camera_system.capture_depth_image()
                world_pos = _camera_system.get_object_world_position(
                    center_pixel[0], center_pixel[1], depth_image
                )
                
                if world_pos:
                    matching_objects.append({
                        'description': f"{obj_color} {obj_shape}",
                        'pixel_position': center_pixel,
                        'world_position': [round(pos, 3) for pos in world_pos]
                    })
        
        if not matching_objects:
            available_objects = [f"{obj['color']} {obj['shape']}" for obj in detected_objects]
            return f"Object '{object_description}' not found. Available objects: {', '.join(available_objects)}"
        
        if len(matching_objects) == 1:
            obj = matching_objects[0]
            return f"Found {obj['description']} at world position {obj['world_position']} (pixel position {obj['pixel_position']})"
        else:
            result = f"Found {len(matching_objects)} matching objects:\n"
            for i, obj in enumerate(matching_objects):
                result += f"{i+1}. {obj['description']} at {obj['world_position']}\n"
            return result
        
    except Exception as e:
        logger.error(f"Error in get_object_position: {e}")
        return f"Error getting object position: {e}"

@tool
def switch_llm_model(provider: str) -> str:
    """Switch between different LLM providers (anthropic, openai, google)"""
    try:
        if _llm_provider_manager is None:
            return "Error: LLM provider manager not initialized"
        
        available_providers = _llm_provider_manager.get_available_providers()
        
        if provider.lower() not in available_providers:
            return f"Provider '{provider}' not available. Available providers: {', '.join(available_providers)}"
        
        success = _llm_provider_manager.switch_provider(provider.lower())
        
        if success:
            current_provider = _llm_provider_manager.current_provider
            capabilities = _llm_provider_manager.get_provider_capabilities(current_provider)
            return f"Successfully switched to {provider}. Current model: {capabilities.get('model', 'Unknown')}"
        else:
            return f"Failed to switch to provider '{provider}'"
        
    except Exception as e:
        logger.error(f"Error in switch_llm_model: {e}")
        return f"Error switching LLM provider: {e}"

@tool
def compare_llm_responses(command: str) -> str:
    """Get responses from all available LLM providers for comparison"""
    try:
        if _llm_provider_manager is None:
            return "Error: LLM provider manager not initialized"
        
        responses = _llm_provider_manager.compare_providers_response(command)
        
        if not responses:
            return "No LLM providers available for comparison"
        
        result = f"LLM Provider Comparison for command: '{command}'\n\n"
        
        for provider, response in responses.items():
            result += f"=== {provider.upper()} ===\n"
            result += f"{response}\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in compare_llm_responses: {e}")
        return f"Error comparing LLM responses: {e}"

# List of all available tools
ROBOT_TOOLS = [
    move_robot_to_position,
    pick_up_object,
    place_object_at_position,
    get_robot_status,
    scan_environment,
    reset_robot_position,
    open_gripper,
    close_gripper,
    get_object_position,
    switch_llm_model,
    compare_llm_responses
]

def get_tools_info() -> Dict[str, Any]:
    """Get information about all available tools."""
    tools_info = {}
    
    for tool in ROBOT_TOOLS:
        tools_info[tool.name] = {
            'name': tool.name,
            'description': tool.description,
            'args': tool.args if hasattr(tool, 'args') else None
        }
    
    return tools_info 
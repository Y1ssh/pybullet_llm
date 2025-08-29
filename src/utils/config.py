"""
Configuration management for PyBullet LLM Robotics project.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RobotConfig:
    """Configuration for robot parameters."""
    robot_type: str = "kuka_iiwa"
    urdf_path: str = "data/robot_models/kuka_iiwa/model.urdf"
    base_position: List[float] = field(default_factory=lambda: [0, 0, 0])
    joint_limits: List[Tuple[float, float]] = field(default_factory=list)
    home_position: List[float] = field(default_factory=lambda: [0, 0, 0, -1.57, 0, 1.57, 0])
    
@dataclass
class LLMConfig:
    """Configuration for LLM providers and parameters."""
    provider: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "anthropic"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL_ANTHROPIC", "claude-3-5-sonnet-20241022"))
    openai_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL_OPENAI", "gpt-4o"))
    google_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL_GOOGLE", "gemini-2.0-flash"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "1000")))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    enable_comparison: bool = field(default_factory=lambda: os.getenv("ENABLE_LLM_COMPARISON", "false").lower() == "true")

@dataclass
class VisionConfig:
    """Configuration for computer vision parameters."""
    camera_width: int = field(default_factory=lambda: int(os.getenv("CAMERA_WIDTH", "640")))
    camera_height: int = field(default_factory=lambda: int(os.getenv("CAMERA_HEIGHT", "480")))
    enable_vision: bool = field(default_factory=lambda: os.getenv("ENABLE_VISION", "true").lower() == "true")

@dataclass
class SimulationConfig:
    """Configuration for physics simulation parameters."""
    timestep: float = field(default_factory=lambda: float(os.getenv("SIMULATION_TIMESTEP", "0.004166667")))
    physics_engine: str = field(default_factory=lambda: os.getenv("PHYSICS_ENGINE", "bullet"))
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

# Global configuration instances
robot_config = RobotConfig()
llm_config = LLMConfig()
vision_config = VisionConfig()
simulation_config = SimulationConfig() 
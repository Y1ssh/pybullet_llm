"""
PyBullet environment setup and management.
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotEnvironment:
    """PyBullet environment for robot simulation with camera and object management."""
    
    def __init__(self, gui_mode: bool = True, timestep: float = 0.004166667):
        """
        Initialize PyBullet environment.
        
        Args:
            gui_mode: Whether to use GUI mode
            timestep: Physics simulation timestep
        """
        self.gui_mode = gui_mode
        self.timestep = timestep
        self.physics_client = None
        self.plane_id = None
        self.table_id = None
        self.objects = {}  # Track spawned objects
        self.camera_params = {}
        self.gravity = [0, 0, -9.81]
        
        # Default camera parameters
        self.default_camera_distance = 1.5
        self.default_camera_yaw = 45
        self.default_camera_pitch = -30
        self.default_camera_target = [0, 0, 0.5]
        
    def setup_physics(self) -> None:
        """Setup PyBullet physics engine and basic environment."""
        try:
            # Connect to PyBullet
            if self.gui_mode:
                self.physics_client = p.connect(p.GUI)
                # Set camera view
                p.resetDebugVisualizerCamera(
                    cameraDistance=self.default_camera_distance,
                    cameraYaw=self.default_camera_yaw,
                    cameraPitch=self.default_camera_pitch,
                    cameraTargetPosition=self.default_camera_target,
                    physicsClientId=self.physics_client
                )
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            # Set additional search path for URDF files
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Configure physics
            p.setGravity(*self.gravity, physicsClientId=self.physics_client)
            p.setTimeStep(self.timestep, physicsClientId=self.physics_client)
            p.setRealTimeSimulation(0, physicsClientId=self.physics_client)
            
            # Load ground plane
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
            
            # Load table (optional)
            self.load_table()
            
            logger.info(f"PyBullet environment initialized (Client ID: {self.physics_client})")
            
        except Exception as e:
            logger.error(f"Failed to setup physics: {e}")
            raise
    
    def load_table(self, position: List[float] = [0, 0, 0], scale: float = 1.0) -> int:
        """
        Load a table into the environment.
        
        Args:
            position: Table position [x, y, z]
            scale: Table scale factor
            
        Returns:
            Table object ID
        """
        try:
            # Try to load table from pybullet_data
            table_urdf = "table/table.urdf"
            
            self.table_id = p.loadURDF(
                table_urdf, 
                basePosition=position,
                globalScaling=scale,
                physicsClientId=self.physics_client
            )
            
            self.objects[self.table_id] = {
                'type': 'table',
                'position': position,
                'scale': scale
            }
            
            logger.info(f"Table loaded with ID: {self.table_id}")
            return self.table_id
            
        except Exception:
            # Create simple table if URDF not available
            logger.warning("Table URDF not found, creating simple table")
            return self.create_simple_table(position, scale)
    
    def create_simple_table(self, position: List[float] = [0, 0, 0], scale: float = 1.0) -> int:
        """
        Create a simple table using basic shapes.
        
        Args:
            position: Table position [x, y, z]
            scale: Table scale factor
            
        Returns:
            Table object ID
        """
        # Table dimensions
        table_height = 0.75 * scale
        table_width = 1.0 * scale
        table_depth = 0.6 * scale
        table_thickness = 0.05 * scale
        
        # Create table top
        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[table_width/2, table_depth/2, table_thickness/2],
            physicsClientId=self.physics_client
        )
        
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[table_width/2, table_depth/2, table_thickness/2],
            rgbaColor=[0.6, 0.4, 0.2, 1],
            physicsClientId=self.physics_client
        )
        
        table_position = [position[0], position[1], position[2] + table_height]
        
        self.table_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=table_position,
            physicsClientId=self.physics_client
        )
        
        self.objects[self.table_id] = {
            'type': 'simple_table',
            'position': table_position,
            'scale': scale
        }
        
        logger.info(f"Simple table created with ID: {self.table_id}")
        return self.table_id
    
    def load_robot(self, urdf_path: str, base_position: List[float] = [0, 0, 0]) -> int:
        """
        Load robot into environment.
        
        Args:
            urdf_path: Path to robot URDF file
            base_position: Robot base position
            
        Returns:
            Robot object ID
        """
        try:
            robot_id = p.loadURDF(
                urdf_path,
                basePosition=base_position,
                physicsClientId=self.physics_client
            )
            
            self.objects[robot_id] = {
                'type': 'robot',
                'urdf_path': urdf_path,
                'position': base_position
            }
            
            logger.info(f"Robot loaded with ID: {robot_id}")
            return robot_id
            
        except Exception as e:
            logger.error(f"Failed to load robot from {urdf_path}: {e}")
            raise
    
    def spawn_object(self, shape: str, position: List[float], color: List[float] = [1, 0, 0, 1], 
                     size: List[float] = [0.05, 0.05, 0.05], mass: float = 0.1) -> int:
        """
        Spawn a simple object in the environment.
        
        Args:
            shape: Object shape ('box', 'sphere', 'cylinder')
            position: Object position [x, y, z]
            color: Object color [r, g, b, a]
            size: Object size parameters
            mass: Object mass
            
        Returns:
            Object ID
        """
        try:
            if shape.lower() == 'box':
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=size,
                    physicsClientId=self.physics_client
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=size,
                    rgbaColor=color,
                    physicsClientId=self.physics_client
                )
            elif shape.lower() == 'sphere':
                radius = size[0] if isinstance(size, list) else size
                collision_shape = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    physicsClientId=self.physics_client
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    rgbaColor=color,
                    physicsClientId=self.physics_client
                )
            elif shape.lower() == 'cylinder':
                radius = size[0] if isinstance(size, list) else size
                height = size[2] if isinstance(size, list) and len(size) > 2 else size
                collision_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    height=height,
                    physicsClientId=self.physics_client
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=color,
                    physicsClientId=self.physics_client
                )
            else:
                raise ValueError(f"Unsupported shape: {shape}")
            
            # Create the object
            object_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                physicsClientId=self.physics_client
            )
            
            # Store object information
            self.objects[object_id] = {
                'type': 'primitive',
                'shape': shape,
                'position': position,
                'color': color,
                'size': size,
                'mass': mass
            }
            
            logger.info(f"Object {shape} spawned with ID: {object_id}")
            return object_id
            
        except Exception as e:
            logger.error(f"Failed to spawn {shape} object: {e}")
            raise
    
    def setup_camera(self, width: int = 640, height: int = 480, 
                     camera_position: List[float] = [1, 1, 1],
                     target_position: List[float] = [0, 0, 0.5]) -> None:
        """
        Setup camera for image capture.
        
        Args:
            width: Image width
            height: Image height
            camera_position: Camera position [x, y, z]
            target_position: Camera target position [x, y, z]
        """
        # Camera parameters
        fov = 60  # Field of view
        aspect = width / height
        near = 0.1
        far = 10.0
        
        # Calculate view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.physics_client
        )
        
        # Calculate projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far,
            physicsClientId=self.physics_client
        )
        
        self.camera_params = {
            'width': width,
            'height': height,
            'view_matrix': view_matrix,
            'projection_matrix': projection_matrix,
            'camera_position': camera_position,
            'target_position': target_position
        }
        
        logger.info(f"Camera setup: {width}x{height}, FOV: {fov}")
    
    def get_camera_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture RGB and depth images from camera.
        
        Returns:
            Tuple of (rgb_image, depth_image)
        """
        if not self.camera_params:
            self.setup_camera()  # Setup default camera if not configured
        
        # Capture image
        width = self.camera_params['width']
        height = self.camera_params['height']
        view_matrix = self.camera_params['view_matrix']
        projection_matrix = self.camera_params['projection_matrix']
        
        camera_data = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            physicsClientId=self.physics_client
        )
        
        # Extract RGB image
        rgb_array = np.array(camera_data[2], dtype=np.uint8)
        rgb_image = rgb_array.reshape((height, width, 4))[:, :, :3]  # Remove alpha channel
        
        # Extract depth image
        depth_buffer = np.array(camera_data[3], dtype=np.float32)
        depth_image = depth_buffer.reshape((height, width))
        
        return rgb_image, depth_image
    
    def step_simulation(self, steps: int = 1) -> None:
        """
        Step the physics simulation.
        
        Args:
            steps: Number of simulation steps
        """
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.physics_client)
    
    def get_object_positions(self) -> Dict[int, List[float]]:
        """
        Get positions of all objects in the environment.
        
        Returns:
            Dictionary mapping object IDs to positions
        """
        positions = {}
        for object_id in self.objects.keys():
            try:
                position, _ = p.getBasePositionAndOrientation(
                    object_id, 
                    physicsClientId=self.physics_client
                )
                positions[object_id] = list(position)
            except Exception as e:
                logger.warning(f"Failed to get position for object {object_id}: {e}")
        
        return positions
    
    def remove_object(self, object_id: int) -> None:
        """
        Remove object from environment.
        
        Args:
            object_id: Object ID to remove
        """
        try:
            p.removeBody(object_id, physicsClientId=self.physics_client)
            if object_id in self.objects:
                del self.objects[object_id]
            logger.info(f"Object {object_id} removed")
        except Exception as e:
            logger.error(f"Failed to remove object {object_id}: {e}")
    
    def get_object_info(self, object_id: int) -> Optional[Dict]:
        """
        Get information about a specific object.
        
        Args:
            object_id: Object ID
            
        Returns:
            Object information dictionary or None if not found
        """
        if object_id in self.objects:
            info = self.objects[object_id].copy()
            try:
                position, orientation = p.getBasePositionAndOrientation(
                    object_id, 
                    physicsClientId=self.physics_client
                )
                info['current_position'] = list(position)
                info['current_orientation'] = list(orientation)
            except Exception as e:
                logger.warning(f"Failed to get current pose for object {object_id}: {e}")
            
            return info
        
        return None
    
    def reset_environment(self) -> None:
        """Reset environment to initial state."""
        # Remove all non-static objects
        objects_to_remove = []
        for object_id, info in self.objects.items():
            if info['type'] not in ['table', 'simple_table'] and object_id != self.plane_id:
                objects_to_remove.append(object_id)
        
        for object_id in objects_to_remove:
            self.remove_object(object_id)
        
        logger.info("Environment reset")
    
    def cleanup(self) -> None:
        """Cleanup and disconnect from PyBullet."""
        if self.physics_client is not None:
            try:
                p.disconnect(physicsClientId=self.physics_client)
                logger.info("PyBullet environment cleaned up")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self.physics_client = None
    
    def get_environment_info(self) -> Dict:
        """
        Get comprehensive environment information.
        
        Returns:
            Dictionary with environment state
        """
        info = {
            'physics_client': self.physics_client,
            'gui_mode': self.gui_mode,
            'timestep': self.timestep,
            'gravity': self.gravity,
            'num_objects': len(self.objects),
            'objects': self.objects.copy(),
            'camera_configured': bool(self.camera_params)
        }
        
        return info
    
    def save_state(self) -> int:
        """
        Save current simulation state.
        
        Returns:
            State ID for restoration
        """
        state_id = p.saveState(physicsClientId=self.physics_client)
        logger.info(f"State saved with ID: {state_id}")
        return state_id
    
    def restore_state(self, state_id: int) -> None:
        """
        Restore simulation to saved state.
        
        Args:
            state_id: State ID to restore
        """
        p.restoreState(state_id, physicsClientId=self.physics_client)
        logger.info(f"State {state_id} restored") 
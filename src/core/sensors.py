"""
Camera and sensor integration for robot perception.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .environment import RobotEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraSystem:
    """Camera system for robot perception and object detection."""
    
    def __init__(self, environment: RobotEnvironment):
        """
        Initialize camera system.
        
        Args:
            environment: Robot environment instance
        """
        self.environment = environment
        self.last_rgb_image = None
        self.last_depth_image = None
        self.calibration_data = {}
        
    def capture_rgb_image(self) -> np.ndarray:
        """
        Capture RGB image from camera.
        
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        try:
            rgb_image, _ = self.environment.get_camera_image()
            self.last_rgb_image = rgb_image
            logger.debug("RGB image captured")
            return rgb_image
        except Exception as e:
            logger.error(f"Failed to capture RGB image: {e}")
            raise
    
    def capture_depth_image(self) -> np.ndarray:
        """
        Capture depth image from camera.
        
        Returns:
            Depth image as numpy array (H, W)
        """
        try:
            _, depth_image = self.environment.get_camera_image()
            self.last_depth_image = depth_image
            logger.debug("Depth image captured")
            return depth_image
        except Exception as e:
            logger.error(f"Failed to capture depth image: {e}")
            raise
    
    def capture_rgbd_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture both RGB and depth images.
        
        Returns:
            Tuple of (rgb_image, depth_image)
        """
        try:
            rgb_image, depth_image = self.environment.get_camera_image()
            self.last_rgb_image = rgb_image
            self.last_depth_image = depth_image
            logger.debug("RGBD image captured")
            return rgb_image, depth_image
        except Exception as e:
            logger.error(f"Failed to capture RGBD image: {e}")
            raise
    
    def get_point_cloud(self, rgb_image: Optional[np.ndarray] = None, 
                       depth_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate point cloud from depth image.
        
        Args:
            rgb_image: RGB image (optional, uses last captured if None)
            depth_image: Depth image (optional, uses last captured if None)
            
        Returns:
            Point cloud as numpy array (N, 6) with [x, y, z, r, g, b]
        """
        if rgb_image is None:
            rgb_image = self.last_rgb_image
        if depth_image is None:
            depth_image = self.last_depth_image
            
        if rgb_image is None or depth_image is None:
            raise ValueError("No RGB or depth image available")
        
        # Get camera parameters
        camera_params = self.environment.camera_params
        if not camera_params:
            raise ValueError("Camera not configured")
        
        height, width = depth_image.shape
        
        # Camera intrinsics (simplified)
        fx = fy = width / 2  # Approximation
        cx, cy = width / 2, height / 2
        
        # Generate point cloud
        points = []
        colors = []
        
        for v in range(0, height, 4):  # Subsample for performance
            for u in range(0, width, 4):
                z = depth_image[v, u]
                if z > 0.1 and z < 5.0:  # Valid depth range
                    # Convert to 3D coordinates
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    
                    points.append([x, y, z])
                    colors.append(rgb_image[v, u] / 255.0)  # Normalize colors
        
        if not points:
            return np.array([]).reshape(0, 6)
        
        points = np.array(points)
        colors = np.array(colors)
        
        # Combine points and colors
        point_cloud = np.hstack([points, colors])
        
        logger.debug(f"Generated point cloud with {len(point_cloud)} points")
        return point_cloud
    
    def detect_objects_in_view(self, rgb_image: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Detect objects in the current camera view using simple computer vision.
        
        Args:
            rgb_image: RGB image (optional, captures new if None)
            
        Returns:
            List of detected objects with properties
        """
        if rgb_image is None:
            rgb_image = self.capture_rgb_image()
        
        # Import OpenCV for object detection
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available for object detection")
            return []
        
        detected_objects = []
        
        try:
            # Convert to HSV for better color detection
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for common objects
            color_ranges = {
                'red': [(0, 50, 50), (10, 255, 255)],
                'green': [(40, 50, 50), (80, 255, 255)],
                'blue': [(100, 50, 50), (130, 255, 255)],
                'yellow': [(20, 50, 50), (40, 255, 255)],
            }
            
            for color_name, (lower, upper) in color_ranges.items():
                # Create mask for color
                lower_bound = np.array(lower)
                upper_bound = np.array(upper)
                mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Filter small objects
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Calculate center
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Estimate object size
                        object_size = max(w, h)
                        
                        detected_objects.append({
                            'color': color_name,
                            'center': (center_x, center_y),
                            'bounding_box': (x, y, w, h),
                            'area': area,
                            'size': object_size,
                            'shape': self._classify_shape(contour)
                        })
            
            logger.debug(f"Detected {len(detected_objects)} objects")
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def _classify_shape(self, contour) -> str:
        """
        Classify the shape of a contour.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            Shape classification string
        """
        try:
            import cv2
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Classify based on number of vertices
            vertices = len(approx)
            
            if vertices == 3:
                return "triangle"
            elif vertices == 4:
                # Check if rectangle or square
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    return "square"
                else:
                    return "rectangle"
            elif vertices > 4:
                # Check circularity
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        return "circle"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    def get_object_world_position(self, pixel_x: int, pixel_y: int, 
                                 depth_image: Optional[np.ndarray] = None) -> Optional[Tuple[float, float, float]]:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            depth_image: Depth image (optional, uses last captured if None)
            
        Returns:
            World coordinates (x, y, z) or None if invalid
        """
        if depth_image is None:
            depth_image = self.last_depth_image
            
        if depth_image is None:
            logger.warning("No depth image available")
            return None
        
        # Get depth at pixel
        if 0 <= pixel_y < depth_image.shape[0] and 0 <= pixel_x < depth_image.shape[1]:
            depth = depth_image[pixel_y, pixel_x]
            
            if depth > 0.1 and depth < 5.0:  # Valid depth range
                # Get camera parameters
                camera_params = self.environment.camera_params
                if not camera_params:
                    logger.warning("Camera not configured")
                    return None
                
                # Camera intrinsics (simplified)
                width = camera_params['width']
                height = camera_params['height']
                fx = fy = width / 2  # Approximation
                cx, cy = width / 2, height / 2
                
                # Convert to camera coordinates
                x_cam = (pixel_x - cx) * depth / fx
                y_cam = (pixel_y - cy) * depth / fy
                z_cam = depth
                
                # Transform to world coordinates (simplified)
                # This would need proper camera extrinsics in a real system
                camera_pos = camera_params['camera_position']
                
                world_x = camera_pos[0] + x_cam
                world_y = camera_pos[1] + y_cam
                world_z = camera_pos[2] - z_cam  # Assuming camera looks down
                
                return (world_x, world_y, world_z)
        
        return None
    
    def calibrate_camera(self, calibration_points: List[Dict]) -> bool:
        """
        Calibrate camera using known world points.
        
        Args:
            calibration_points: List of {pixel: (u, v), world: (x, y, z)} points
            
        Returns:
            True if calibration successful
        """
        try:
            # Simple calibration - store calibration data
            self.calibration_data = {
                'points': calibration_points,
                'calibrated': True
            }
            
            logger.info(f"Camera calibrated with {len(calibration_points)} points")
            return True
            
        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            return False
    
    def save_image(self, filename: str, image_type: str = 'rgb') -> bool:
        """
        Save captured image to file.
        
        Args:
            filename: Output filename
            image_type: Type of image to save ('rgb' or 'depth')
            
        Returns:
            True if save successful
        """
        try:
            import cv2
            
            if image_type.lower() == 'rgb':
                if self.last_rgb_image is not None:
                    # Convert RGB to BGR for OpenCV
                    bgr_image = cv2.cvtColor(self.last_rgb_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, bgr_image)
                    logger.info(f"RGB image saved to {filename}")
                    return True
                else:
                    logger.warning("No RGB image to save")
                    return False
                    
            elif image_type.lower() == 'depth':
                if self.last_depth_image is not None:
                    # Normalize depth for visualization
                    depth_normalized = (self.last_depth_image * 255).astype(np.uint8)
                    cv2.imwrite(filename, depth_normalized)
                    logger.info(f"Depth image saved to {filename}")
                    return True
                else:
                    logger.warning("No depth image to save")
                    return False
            else:
                logger.error(f"Unknown image type: {image_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    def get_camera_info(self) -> Dict:
        """
        Get camera system information.
        
        Returns:
            Dictionary with camera information
        """
        info = {
            'camera_configured': bool(self.environment.camera_params),
            'last_rgb_shape': self.last_rgb_image.shape if self.last_rgb_image is not None else None,
            'last_depth_shape': self.last_depth_image.shape if self.last_depth_image is not None else None,
            'calibrated': self.calibration_data.get('calibrated', False),
            'calibration_points': len(self.calibration_data.get('points', []))
        }
        
        if self.environment.camera_params:
            info.update(self.environment.camera_params)
        
        return info 
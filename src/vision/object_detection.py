"""
Object detection algorithms for robot perception.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """Object detection using computer vision techniques."""
    
    def __init__(self):
        """Initialize object detector."""
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (40, 255, 255)],
            'orange': [(10, 50, 50), (20, 255, 255)],
            'purple': [(130, 50, 50), (160, 255, 255)],
            'cyan': [(80, 50, 50), (100, 255, 255)],
            'pink': [(160, 50, 50), (170, 255, 255)]
        }
        
        # Minimum contour area for object detection
        self.min_contour_area = 100
        
        # Shape classification parameters
        self.shape_epsilon_factor = 0.02
        
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all objects in the image.
        
        Args:
            image: Input RGB image
            
        Returns:
            List of detected objects with properties
        """
        try:
            detected_objects = []
            
            # Convert to HSV for better color detection
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Detect objects by color
            for color_name in self.color_ranges.keys():
                color_objects = self.detect_by_color(image, color_name)
                detected_objects.extend(color_objects)
            
            # Remove duplicates and merge overlapping detections
            detected_objects = self._merge_overlapping_detections(detected_objects)
            
            # Sort by area (largest first)
            detected_objects.sort(key=lambda x: x['area'], reverse=True)
            
            logger.debug(f"Detected {len(detected_objects)} objects")
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def detect_by_color(self, image: np.ndarray, color_name: str) -> List[Dict]:
        """
        Detect objects of a specific color.
        
        Args:
            image: Input RGB image
            color_name: Name of color to detect
            
        Returns:
            List of detected objects of the specified color
        """
        try:
            if color_name not in self.color_ranges:
                logger.warning(f"Unknown color: {color_name}")
                return []
            
            # Convert to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Create mask for the color
            mask = self.create_color_mask(image, color_name)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > self.min_contour_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate centroid
                    centroid = self.get_object_centroid(contour)
                    
                    # Classify shape
                    shape = self.classify_shape(contour)
                    
                    # Calculate additional properties
                    perimeter = cv2.arcLength(contour, True)
                    aspect_ratio = w / h if h > 0 else 0
                    extent = area / (w * h) if w > 0 and h > 0 else 0
                    
                    detected_object = {
                        'color': color_name,
                        'shape': shape,
                        'center': centroid,
                        'bounding_box': (x, y, w, h),
                        'area': area,
                        'perimeter': perimeter,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'contour': contour,
                        'confidence': self._calculate_detection_confidence(contour, mask, x, y, w, h)
                    }
                    
                    detected_objects.append(detected_object)
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error detecting {color_name} objects: {e}")
            return []
    
    def detect_by_shape(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects by shape analysis.
        
        Args:
            image: Input RGB image
            
        Returns:
            List of detected objects with shape classification
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > self.min_contour_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate centroid
                    centroid = self.get_object_centroid(contour)
                    
                    # Classify shape
                    shape = self.classify_shape(contour)
                    
                    detected_object = {
                        'color': 'unknown',
                        'shape': shape,
                        'center': centroid,
                        'bounding_box': (x, y, w, h),
                        'area': area,
                        'contour': contour
                    }
                    
                    detected_objects.append(detected_object)
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in shape-based detection: {e}")
            return []
    
    def get_object_centroid(self, contour: np.ndarray) -> Tuple[int, int]:
        """
        Calculate centroid of object contour.
        
        Args:
            contour: Object contour
            
        Returns:
            Centroid coordinates (x, y)
        """
        try:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback to bounding box center
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
            
            return (cx, cy)
            
        except Exception as e:
            logger.error(f"Error calculating centroid: {e}")
            return (0, 0)
    
    def classify_shape(self, contour: np.ndarray) -> str:
        """
        Classify the shape of an object contour.
        
        Args:
            contour: Object contour
            
        Returns:
            Shape classification string
        """
        try:
            # Approximate contour
            epsilon = self.shape_epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get number of vertices
            vertices = len(approx)
            
            # Basic shape classification
            if vertices == 3:
                return "triangle"
            elif vertices == 4:
                # Check if it's a square or rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                if 0.95 <= aspect_ratio <= 1.05:
                    return "square"
                else:
                    return "rectangle"
            elif vertices == 5:
                return "pentagon"
            elif vertices == 6:
                return "hexagon"
            elif vertices > 6:
                # Check if it's circular
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        return "circle"
                    else:
                        return "ellipse"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error classifying shape: {e}")
            return "unknown"
    
    def create_color_mask(self, image: np.ndarray, color_name: str) -> np.ndarray:
        """
        Create a color mask for the specified color.
        
        Args:
            image: Input RGB image
            color_name: Name of color
            
        Returns:
            Binary mask for the color
        """
        try:
            if color_name not in self.color_ranges:
                return np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Convert to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            color_data = self.color_ranges[color_name]
            
            if len(color_data) == 4:  # Red has two ranges
                lower1, upper1, lower2, upper2 = color_data
                mask1 = cv2.inRange(hsv_image, np.array(lower1), np.array(upper1))
                mask2 = cv2.inRange(hsv_image, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower, upper = color_data
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error creating color mask for {color_name}: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _calculate_detection_confidence(self, contour: np.ndarray, mask: np.ndarray, 
                                      x: int, y: int, w: int, h: int) -> float:
        """
        Calculate confidence score for object detection.
        
        Args:
            contour: Object contour
            mask: Color mask
            x, y, w, h: Bounding box coordinates
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Calculate area ratio
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            area_ratio = contour_area / bbox_area if bbox_area > 0 else 0
            
            # Calculate mask coverage in bounding box
            roi_mask = mask[y:y+h, x:x+w]
            mask_pixels = np.sum(roi_mask > 0)
            total_pixels = w * h
            coverage_ratio = mask_pixels / total_pixels if total_pixels > 0 else 0
            
            # Calculate perimeter-area ratio (compactness)
            perimeter = cv2.arcLength(contour, True)
            compactness = (4 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Combine metrics for confidence
            confidence = (area_ratio * 0.4 + coverage_ratio * 0.4 + compactness * 0.2)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _merge_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Merge overlapping detections to reduce duplicates.
        
        Args:
            detections: List of detected objects
            
        Returns:
            List of merged detections
        """
        try:
            if len(detections) <= 1:
                return detections
            
            merged = []
            used = set()
            
            for i, det1 in enumerate(detections):
                if i in used:
                    continue
                
                # Check for overlaps with other detections
                overlapping = [det1]
                
                for j, det2 in enumerate(detections[i+1:], i+1):
                    if j in used:
                        continue
                    
                    if self._boxes_overlap(det1['bounding_box'], det2['bounding_box']):
                        overlapping.append(det2)
                        used.add(j)
                
                # Merge overlapping detections
                if len(overlapping) == 1:
                    merged.append(det1)
                else:
                    merged_detection = self._merge_detections(overlapping)
                    merged.append(merged_detection)
                
                used.add(i)
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging detections: {e}")
            return detections
    
    def _boxes_overlap(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> bool:
        """
        Check if two bounding boxes overlap.
        
        Args:
            box1: First bounding box (x, y, w, h)
            box2: Second bounding box (x, y, w, h)
            
        Returns:
            True if boxes overlap
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate overlap
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - overlap_area
        
        # Check if overlap is significant
        overlap_ratio = overlap_area / union_area if union_area > 0 else 0
        
        return overlap_ratio > 0.3  # 30% overlap threshold
    
    def _merge_detections(self, detections: List[Dict]) -> Dict:
        """
        Merge multiple overlapping detections into one.
        
        Args:
            detections: List of overlapping detections
            
        Returns:
            Merged detection
        """
        try:
            # Find detection with highest confidence
            best_detection = max(detections, key=lambda x: x.get('confidence', 0))
            
            # Calculate merged bounding box
            min_x = min(det['bounding_box'][0] for det in detections)
            min_y = min(det['bounding_box'][1] for det in detections)
            max_x = max(det['bounding_box'][0] + det['bounding_box'][2] for det in detections)
            max_y = max(det['bounding_box'][1] + det['bounding_box'][3] for det in detections)
            
            merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            # Create merged detection
            merged = best_detection.copy()
            merged['bounding_box'] = merged_bbox
            merged['center'] = (min_x + (max_x - min_x) // 2, min_y + (max_y - min_y) // 2)
            merged['area'] = sum(det['area'] for det in detections) / len(detections)
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging detections: {e}")
            return detections[0] if detections else {}
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detected objects on the image.
        
        Args:
            image: Input image
            detections: List of detected objects
            
        Returns:
            Image with detection visualizations
        """
        try:
            vis_image = image.copy()
            
            for i, detection in enumerate(detections):
                x, y, w, h = detection['bounding_box']
                center = detection['center']
                color = detection['color']
                shape = detection['shape']
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(vis_image, center, 5, (255, 0, 0), -1)
                
                # Add label
                label = f"{color} {shape}"
                cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                
                # Add object number
                cv2.putText(vis_image, str(i + 1), (center[0] - 10, center[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error visualizing detections: {e}")
            return image
    
    def set_detection_parameters(self, min_area: int = None, epsilon_factor: float = None):
        """
        Set detection parameters.
        
        Args:
            min_area: Minimum contour area for detection
            epsilon_factor: Shape approximation epsilon factor
        """
        if min_area is not None:
            self.min_contour_area = min_area
        
        if epsilon_factor is not None:
            self.shape_epsilon_factor = epsilon_factor
        
        logger.info(f"Detection parameters updated: min_area={self.min_contour_area}, "
                   f"epsilon_factor={self.shape_epsilon_factor}")
    
    def add_color_range(self, color_name: str, hsv_range: Tuple):
        """
        Add a custom color range for detection.
        
        Args:
            color_name: Name of the color
            hsv_range: HSV range tuple(s)
        """
        self.color_ranges[color_name] = hsv_range
        logger.info(f"Added color range for {color_name}")
    
    def get_detection_statistics(self, detections: List[Dict]) -> Dict:
        """
        Get statistics about detected objects.
        
        Args:
            detections: List of detected objects
            
        Returns:
            Statistics dictionary
        """
        if not detections:
            return {"total_objects": 0}
        
        # Count by color
        color_counts = {}
        for det in detections:
            color = det['color']
            color_counts[color] = color_counts.get(color, 0) + 1
        
        # Count by shape
        shape_counts = {}
        for det in detections:
            shape = det['shape']
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        # Calculate average properties
        areas = [det['area'] for det in detections]
        confidences = [det.get('confidence', 0) for det in detections]
        
        stats = {
            "total_objects": len(detections),
            "colors": color_counts,
            "shapes": shape_counts,
            "average_area": np.mean(areas),
            "total_area": sum(areas),
            "average_confidence": np.mean(confidences),
            "min_confidence": min(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0
        }
        
        return stats 
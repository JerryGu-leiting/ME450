"""
Marker-Based Footprint Analysis System - IMPROVED VERSION
Handles noisy backgrounds and various marker sizes
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation
from sklearn.linear_model import RANSACRegressor
import glob
import json
import os


# ============================================================================
# CAMERA CALIBRATION (Same as before)
# ============================================================================

class CameraCalibrator:
    """Handle camera calibration using checkerboard pattern"""
    
    def __init__(self, checkerboard_size=(9, 6), square_size_cm=3.0):
        self.checkerboard_size = checkerboard_size
        self.square_size_cm = square_size_cm
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def calibrate_from_images(self, image_folder):
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), 
                        np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 
                               0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size_cm
        
        objpoints = []
        imgpoints = []
        
        images = glob.glob(os.path.join(image_folder, '*.jpg')) + \
                 glob.glob(os.path.join(image_folder, '*.png'))
        
        if len(images) < 10:
            print(f"Warning: Only {len(images)} calibration images found. Need at least 10-15.")
        
        print(f"Processing {len(images)} calibration images...")
        found_count = 0
        
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                found_count += 1
                objpoints.append(objp)
                
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_refined)
                
                print(f"  ✓ {os.path.basename(fname)}")
            else:
                print(f"  ✗ {os.path.basename(fname)} - checkerboard not found")
        
        if found_count < 10:
            raise ValueError(f"Only {found_count} valid images. Need at least 10.")
        
        print("\nCalibrating camera...")
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                              self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(objpoints)
        print(f"Calibration complete! Mean reprojection error: {mean_error:.4f} pixels")
        
        return self.camera_matrix, self.dist_coeffs
    
    def save_calibration(self, filepath='camera_calibration.json'):
        data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'checkerboard_size': self.checkerboard_size,
            'square_size_cm': self.square_size_cm
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath='camera_calibration.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['dist_coeffs'])
        print(f"Calibration loaded from {filepath}")
        return self.camera_matrix, self.dist_coeffs


# ============================================================================
# IMPROVED MARKER ANALYZER
# ============================================================================

class MarkerAnalyzer:
    """Detect and analyze markers with improved noise rejection"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None, pixels_per_cm=None, 
                 marker_color='black'):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pixels_per_cm = pixels_per_cm
        self.marker_color = marker_color
        
        # IMPROVED: More restrictive default parameters
        self.params = {
            'gaussian_blur': 5,  # Stronger blur to remove noise
            'binary_threshold': 100,  # Lower threshold for black markers
            'adaptive_block_size': 51,  # Use adaptive thresholding
            'adaptive_C': 15,
            'min_area': 100,  # INCREASED from 10 - ignore tiny noise
            'max_area': 5000,  # Reasonable max for markers
            'min_circularity': 0.5,  # More lenient for oval markers
            'min_aspect_ratio': 0.3,  # Filter very elongated shapes
            'max_aspect_ratio': 3.0,
            'use_adaptive': True,  # Use adaptive thresholding by default
        }
        
        # HSV color ranges
        self.color_ranges = {
            'blue': {
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            'red': {
                'lower1': np.array([0, 50, 50]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([170, 50, 50]),
                'upper2': np.array([180, 255, 255])
            },
            'green': {
                'lower': np.array([40, 40, 40]),
                'upper': np.array([80, 255, 255])
            },
            'black': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 80])  # Increased from 50
            },
            'white': {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 30, 255])
            }
        }
    
    def set_parameters(self, **kwargs):
        """Allow manual parameter adjustment"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                print(f"Parameter '{key}' set to {value}")
            else:
                print(f"Warning: Unknown parameter '{key}'")
    
    def set_custom_color_range(self, lower_hsv, upper_hsv):
        self.color_ranges['custom'] = {
            'lower': np.array(lower_hsv),
            'upper': np.array(upper_hsv)
        }
        self.marker_color = 'custom'
        print(f"Custom color range set: {lower_hsv} to {upper_hsv}")
    
    def set_scale_from_reference(self, image, reference_length_cm, point1, point2):
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        self.pixels_per_cm = pixel_distance / reference_length_cm
        
        marker_diameter_cm = 0.3  # 3mm default
        expected_marker_diameter_pixels = marker_diameter_cm * self.pixels_per_cm
        expected_marker_area = np.pi * (expected_marker_diameter_pixels / 2) ** 2
        
        print(f"Scale set: {self.pixels_per_cm:.2f} pixels/cm")
        print(f"Expected marker diameter: {expected_marker_diameter_pixels:.1f} pixels")
        print(f"Expected marker area: {expected_marker_area:.1f} pixels²")
        
        # IMPROVED: Better area range calculation
        self.params['min_area'] = max(50, int(expected_marker_area * 0.3))
        self.params['max_area'] = int(expected_marker_area * 5.0)
        
        print(f"Auto-adjusted area range: {self.params['min_area']} - {self.params['max_area']} pixels²")
        
        return self.pixels_per_cm
    
    def preprocess_image(self, image):
        """
        IMPROVED preprocessing with better noise rejection
        """
        # Undistort if calibration available
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        
        # Convert to grayscale for black markers
        if self.marker_color == 'black':
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # IMPROVED: Stronger denoising
            gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Gaussian blur
            blurred = cv2.GaussianBlur(gray, (self.params['gaussian_blur'], 
                                               self.params['gaussian_blur']), 0)
            
            # IMPROVED: Use adaptive thresholding for textured backgrounds
            if self.params['use_adaptive']:
                binary = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV,
                    self.params['adaptive_block_size'], 
                    self.params['adaptive_C']
                )
            else:
                _, binary = cv2.threshold(blurred, self.params['binary_threshold'], 
                                         255, cv2.THRESH_BINARY_INV)
            
            # IMPROVED: Morphological operations to remove small noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
            
            # Close small holes in markers
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=1)
            
        else:  # Color detection (HSV)
            blurred = cv2.GaussianBlur(image, (self.params['gaussian_blur'], 
                                               self.params['gaussian_blur']), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            if self.marker_color not in self.color_ranges:
                raise ValueError(f"Unknown color: {self.marker_color}")
            
            color_range = self.color_ranges[self.marker_color]
            
            if self.marker_color == 'red':
                mask1 = cv2.inRange(hsv, color_range['lower1'], color_range['upper1'])
                mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
                binary = cv2.bitwise_or(mask1, mask2)
            else:
                binary = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return binary
    
    def detect_markers_contour(self, binary_image, debug=False):
        """
        IMPROVED contour detection with better filtering
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        marker_centers = []
        marker_contours = []
        rejected_reasons = []
        
        print(f"\nAnalyzing {len(contours)} contours...")
        
        for i, contour in enumerate(contours):
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.params['min_area']:
                rejected_reasons.append(f"Contour {i}: area {area:.1f} < min {self.params['min_area']}")
                continue
            if area > self.params['max_area']:
                rejected_reasons.append(f"Contour {i}: area {area:.1f} > max {self.params['max_area']}")
                continue
            
            # Calculate perimeter and circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                rejected_reasons.append(f"Contour {i}: zero perimeter")
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Filter by circularity
            if circularity < self.params['min_circularity']:
                rejected_reasons.append(f"Contour {i}: circularity {circularity:.2f} < {self.params['min_circularity']}")
                continue
            
            # IMPROVED: Check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            if aspect_ratio < self.params['min_aspect_ratio'] or \
               aspect_ratio > self.params['max_aspect_ratio']:
                rejected_reasons.append(f"Contour {i}: aspect ratio {aspect_ratio:.2f} out of range")
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                rejected_reasons.append(f"Contour {i}: zero moment")
                continue
            
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
            marker_centers.append((cx, cy))
            marker_contours.append(contour)
            
            if debug:
                print(f"  ✓ Contour {i}: area={area:.1f}, circ={circularity:.2f}, AR={aspect_ratio:.2f}")
        
        print(f"Detected {len(marker_centers)} valid markers (rejected {len(contours) - len(marker_centers)})")
        
        if debug and len(marker_centers) == 0:
            print("\nRejection reasons:")
            for reason in rejected_reasons[:10]:  # Show first 10
                print(f"  - {reason}")
            if len(rejected_reasons) > 10:
                print(f"  ... and {len(rejected_reasons) - 10} more")
        
        return marker_centers, marker_contours
    
    def pixels_to_world(self, pixel_coords):
        if self.pixels_per_cm is None:
            raise ValueError("Scale not set! Use set_scale_from_reference() first.")
        
        world_coords = []
        for px, py in pixel_coords:
            wx = px / self.pixels_per_cm
            wy = py / self.pixels_per_cm
            world_coords.append((wx, wy))
        
        return world_coords
    
    def calculate_step_width(self, world_coords, method='gaitline'):
        if len(world_coords) < 3:
            raise ValueError("Need at least 3 markers for analysis")
        
        coords_array = np.array(world_coords)
        
        if method == 'gaitline':
            X = coords_array[:, 0].reshape(-1, 1)
            y = coords_array[:, 1]
            
            ransac = RANSACRegressor(random_state=42, residual_threshold=2.0)
            ransac.fit(X, y)
            
            distances = []
            for i, (x_val, y_val) in enumerate(coords_array):
                pred_y = ransac.predict([[x_val]])[0]
                dist = abs(y_val - pred_y)
                distances.append(dist)
            
            median_y = np.median(coords_array[:, 1])
            left_distances = [d for i, d in enumerate(distances) 
                            if coords_array[i, 1] < median_y]
            right_distances = [d for i, d in enumerate(distances) 
                             if coords_array[i, 1] >= median_y]
            
            if left_distances and right_distances:
                min_len = min(len(left_distances), len(right_distances))
                step_widths = [left_distances[i] + right_distances[i] 
                              for i in range(min_len)]
            else:
                step_widths = distances
            
        else:  # pairwise
            sorted_coords = sorted(coords_array, key=lambda p: p[0])
            sorted_array = np.array(sorted_coords)
            
            step_widths = []
            for i in range(len(sorted_array) - 1):
                lateral_distance = abs(sorted_array[i+1, 1] - sorted_array[i, 1])
                step_widths.append(lateral_distance)
        
        step_widths = np.array(step_widths)
        
        # Remove outliers
        if len(step_widths) > 3:
            q1, q3 = np.percentile(step_widths, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            step_widths = step_widths[(step_widths >= lower_bound) & 
                                     (step_widths <= upper_bound)]
        
        results = {
            'step_widths': step_widths.tolist(),
            'mean_cm': np.mean(step_widths),
            'std_cm': np.std(step_widths),
            'min_cm': np.min(step_widths),
            'max_cm': np.max(step_widths),
            'cov_percent': (np.std(step_widths) / np.mean(step_widths) * 100) 
                          if np.mean(step_widths) > 0 else 0,
            'num_steps': len(step_widths)
        }
        
        return results
    
    def visualize_results(self, image, marker_centers, world_coords, 
                         step_width_results, binary_image=None, 
                         all_contours=None, save_path=None):
        """IMPROVED visualization with debug info"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Original image with ALL contours and valid markers
        img_debug = image.copy()
        
        # Draw ALL contours in red (rejected ones)
        if all_contours is not None:
            cv2.drawContours(img_debug, all_contours, -1, (0, 0, 255), 1)
        
        # Draw valid markers in green
        for cx, cy in marker_centers:
            cv2.circle(img_debug, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.circle(img_debug, (int(cx), int(cy)), 20, (0, 255, 0), 2)
            cv2.line(img_debug, (int(cx)-15, int(cy)), (int(cx)+15, int(cy)), 
                    (255, 0, 0), 2)
            cv2.line(img_debug, (int(cx), int(cy)-15), (int(cx), int(cy)+15), 
                    (255, 0, 0), 2)
        
        axes[0, 0].imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Detected Markers (n={len(marker_centers)}) - Green = Valid, Red = Rejected', 
                            fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Binary image
        if binary_image is not None:
            axes[0, 1].imshow(binary_image, cmap='gray')
            axes[0, 1].set_title('Binary Image (After Preprocessing)', 
                                fontsize=11, fontweight='bold')
            axes[0, 1].axis('off')
        
        # 3. World coordinates
        if world_coords and len(world_coords) >= 2:
            wc_array = np.array(world_coords)
            axes[1, 0].scatter(wc_array[:, 0], wc_array[:, 1], 
                             c='blue', s=200, alpha=0.6, edgecolors='black', linewidths=2)
            
            for i, (x, y) in enumerate(world_coords):
                axes[1, 0].annotate(f'{i+1}', (x, y), fontsize=10, 
                                   ha='center', va='center', color='white', weight='bold')
            
            if len(world_coords) >= 2:
                X = wc_array[:, 0].reshape(-1, 1)
                y = wc_array[:, 1]
                ransac = RANSACRegressor(random_state=42, residual_threshold=2.0)
                ransac.fit(X, y)
                x_line = np.linspace(wc_array[:, 0].min(), wc_array[:, 0].max(), 100)
                y_line = ransac.predict(x_line.reshape(-1, 1))
                axes[1, 0].plot(x_line, y_line, 'r--', linewidth=2, label='Gait line')
                
                for x_val, y_val in world_coords:
                    pred_y = ransac.predict([[x_val]])[0]
                    axes[1, 0].plot([x_val, x_val], [y_val, pred_y], 
                                  'gray', alpha=0.3, linestyle=':')
            
            axes[1, 0].set_xlabel('X (cm)', fontsize=10)
            axes[1, 0].set_ylabel('Y (cm)', fontsize=10)
            axes[1, 0].set_title('Marker Positions (World Coordinates)', 
                                fontsize=11, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_aspect('equal')
        
        # 4. Step width distribution
        step_widths = step_width_results['step_widths']
        if len(step_widths) > 0:
            axes[1, 1].hist(step_widths, bins=min(10, max(3, len(step_widths))), 
                           color='skyblue', edgecolor='black', alpha=0.7)
            axes[1, 1].axvline(step_width_results['mean_cm'], color='red', 
                              linestyle='--', linewidth=2, 
                              label=f"Mean: {step_width_results['mean_cm']:.2f} cm")
            axes[1, 1].axvline(step_width_results['mean_cm'] - step_width_results['std_cm'], 
                              color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            axes[1, 1].axvline(step_width_results['mean_cm'] + step_width_results['std_cm'], 
                              color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                              label=f"±1 SD: {step_width_results['std_cm']:.2f} cm")
            
            axes[1, 1].set_xlabel('Step Width (cm)', fontsize=10)
            axes[1, 1].set_ylabel('Frequency', fontsize=10)
            axes[1, 1].set_title('Step Width Distribution', fontsize=11, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            summary_text = f"""
n = {step_width_results['num_steps']} steps
Mean = {step_width_results['mean_cm']:.2f} cm
SD = {step_width_results['std_cm']:.2f} cm
CoV = {step_width_results['cov_percent']:.1f}%
Range = {step_width_results['min_cm']:.2f} - {step_width_results['max_cm']:.2f} cm
            """
            axes[1, 1].text(0.98, 0.97, summary_text.strip(), 
                           transform=axes[1, 1].transAxes,
                           fontsize=9, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main_pipeline(image_path, calibration_file=None, reference_points=None, 
                 reference_length_cm=None, marker_color='black', 
                 detection_method='contour', visualize=True, tune_mode=False):
    """Complete pipeline with improved debugging"""
    
    print("="*60)
    print("MARKER-BASED STEP WIDTH ANALYSIS (IMPROVED)")
    print(f"{marker_color.capitalize()} circular markers")
    print("="*60)
    
    # Load image
    print(f"\n1. Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Initialize analyzer
    analyzer = MarkerAnalyzer(marker_color=marker_color)
    
    # Load calibration if available
    if calibration_file and os.path.exists(calibration_file):
        print(f"\n2. Loading camera calibration from {calibration_file}")
        calibrator = CameraCalibrator()
        camera_matrix, dist_coeffs = calibrator.load_calibration(calibration_file)
        analyzer.camera_matrix = camera_matrix
        analyzer.dist_coeffs = dist_coeffs
    else:
        print("\n2. No camera calibration (skipping distortion correction)")
    
    # Set scale
    if reference_points and reference_length_cm:
        print(f"\n3. Setting scale from reference points")
        analyzer.set_scale_from_reference(image, reference_length_cm, 
                                         reference_points[0], reference_points[1])
    else:
        print("\n3. WARNING: No scale reference provided!")
        print("   Estimating scale from image size...")
        # Estimate based on typical A4 paper width (21cm)
        estimated_width_cm = 21.0
        analyzer.pixels_per_cm = image.shape[1] / estimated_width_cm
        print(f"   Estimated: {analyzer.pixels_per_cm:.2f} pixels/cm (may be inaccurate)")
        
        # Set reasonable area limits
        analyzer.params['min_area'] = 100
        analyzer.params['max_area'] = 5000
    
    # Preprocess
    print(f"\n4. Preprocessing image for {marker_color} markers...")
    binary = analyzer.preprocess_image(image)
    
    if tune_mode:
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary Image (White = Detected Regions)')
        plt.axis('off')
        
        # Show size distribution of all contours
        contours_all, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours_all]
        
        plt.subplot(1, 3, 3)
        plt.hist(areas, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.axvline(analyzer.params['min_area'], color='red', linestyle='--', 
                   label=f"Min area: {analyzer.params['min_area']}")
        plt.axvline(analyzer.params['max_area'], color='red', linestyle='--', 
                   label=f"Max area: {analyzer.params['max_area']}")
        plt.xlabel('Contour Area (pixels²)')
        plt.ylabel('Count')
        plt.title(f'Area Distribution (Total: {len(areas)} contours)')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nContour statistics:")
        print(f"  Total contours: {len(areas)}")
        print(f"  Area range: {min(areas):.1f} - {max(areas):.1f} pixels²")
        print(f"  Mean area: {np.mean(areas):.1f} pixels²")
        print(f"  Median area: {np.median(areas):.1f} pixels²")
    
    # Detect markers
    print(f"\n5. Detecting {marker_color} markers...")
    
    # Get ALL contours for visualization
    all_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
    
    marker_centers, marker_contours = analyzer.detect_markers_contour(binary, debug=True)
    
    if len(marker_centers) == 0:
        print(f"\n   ERROR: No markers detected!")
        print("\n   Suggestions:")
        print(f"   1. Run with tune_mode=True to see preprocessing")
        print(f"   2. Try adjusting min_area (current: {analyzer.params['min_area']})")
        print(f"   3. Try adjusting adaptive_C (current: {analyzer.params['adaptive_C']})")
        print(f"   4. Check marker color setting (current: {marker_color})")
        return None
    
    # Convert to world coordinates
    print("\n6. Converting to world coordinates...")
    world_coords = analyzer.pixels_to_world(marker_centers)
    
    # Calculate step width
    print("\n7. Calculating step width and variability...")
    results = analyzer.calculate_step_width(world_coords, method='gaitline')
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Marker Color:        {marker_color.capitalize()}")
    print(f"Number of Markers:   {len(marker_centers)}")
    print(f"Number of Steps:     {results['num_steps']}")
    print(f"Mean Step Width:     {results['mean_cm']:.2f} ± {results['std_cm']:.2f} cm")
    print(f"Variability (CoV):   {results['cov_percent']:.2f} %")
    print(f"Range:               {results['min_cm']:.2f} - {results['max_cm']:.2f} cm")
    print("="*60)
    
    # Visualize
    if visualize:
        print("\n8. Generating visualization...")
        save_path = image_path.rsplit('.', 1)[0] + '_analysis.' + image_path.rsplit('.', 1)[1]
        analyzer.visualize_results(image, marker_centers, world_coords, 
                                  results, binary_image=binary, 
                                  all_contours=all_contours, save_path=save_path)
    
    return results


# ============================================================================
# INTERACTIVE REFERENCE POINT SELECTOR
# ============================================================================

def select_reference_points(image_path):
    """Interactive tool to select reference points"""
    print("\nINTERACTIVE REFERENCE POINT SELECTION")
    print("Click two points of known distance, then press any key")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    max_display_width = 1200
    if image.shape[1] > max_display_width:
        scale = max_display_width / image.shape[1]
        display_image = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        scale = 1.0
        display_image = image.copy()
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            original_x = int(x / scale)
            original_y = int(y / scale)
            points.append((original_x, original_y))
            
            cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_image, f"P{len(points)}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if len(points) == 2:
                cv2.line(display_image, 
                        (int(points[0][0]*scale), int(points[0][1]*scale)),
                        (int(points[1][0]*scale), int(points[1][1]*scale)),
                        (0, 255, 0), 2)
                pixel_dist = np.sqrt((points[1][0]-points[0][0])**2 + 
                                    (points[1][1]-points[0][1])**2)
                cv2.putText(display_image, f"{pixel_dist:.1f} pixels", 
                           (int((points[0][0]+points[1][0])*scale/2), 
                            int((points[0][1]+points[1][1])*scale/2)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Select Reference Points', display_image)
    
    cv2.namedWindow('Select Reference Points')
    cv2.setMouseCallback('Select Reference Points', mouse_callback)
    cv2.imshow('Select Reference Points', display_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) != 2:
        raise ValueError("Must select exactly 2 points")
    
    print(f"\nSelected points:")
    print(f"  Point 1: {points[0]}")
    print(f"  Point 2: {points[1]}")
    pixel_distance = np.sqrt((points[1][0]-points[0][0])**2 + 
                             (points[1][1]-points[0][1])**2)
    print(f"  Distance: {pixel_distance:.1f} pixels")
    
    return tuple(points)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Path to your image
    image_path = 'markers.jpg'  # Change to your image filename
    
    # OPTION 1: Interactive reference point selection (RECOMMENDED)
    print("Select two points of known distance (e.g., corners of paper)")
    reference_points = select_reference_points(image_path)
    reference_length_cm = float(input("\nEnter the distance between these points in cm: "))
    
    # OPTION 2: Manual specification (if you know coordinates)
    # reference_points = ((100, 100), (500, 100))
    # reference_length_cm = 10.0
    
    # Run analysis
    results = main_pipeline(
        image_path=image_path,
        calibration_file=None,
        reference_points=reference_points,
        reference_length_cm=reference_length_cm,
        marker_color='black',  # Your markers are black
        detection_method='contour',
        visualize=True,
        tune_mode=True  # Set to True to see debugging info
    )
    
    # If you need to manually adjust parameters:
    """
    analyzer = MarkerAnalyzer(marker_color='black')
    analyzer.set_parameters(
        min_area=200,  # Increase if too many small detections
        max_area=3000,  # Decrease if detecting large noise
        adaptive_C=10,  # Adjust threshold sensitivity
        min_circularity=0.4  # Lower for oval markers
    )
    """
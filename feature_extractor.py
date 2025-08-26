# Feature Extraction Module
# feature_extractor.py

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
import matplotlib.pyplot as plt

class FeatureExtractor:
    """Extract multiple features from plant leaf images"""
    
    def __init__(self):
        self.target_size = (256, 256)
    
    def preprocess_image(self, image):
        """Preprocess image to standard format"""
        # Convert to RGB if needed
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize image
        image_resized = cv2.resize(image_rgb, self.target_size)
        
        return image_resized
    
    def extract_color_histogram(self, image):
        """Extract color histogram features from HSV color space"""
        # Convert RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv_image], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [50], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [50], [0, 256])
        
        # Flatten and concatenate
        color_features = np.concatenate([
            hist_h.flatten(),
            hist_s.flatten(),
            hist_v.flatten()
        ])
        
        return color_features
    
    def extract_hu_moments(self, image):
        """Extract Hu moments for shape description"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate moments
        moments = cv2.moments(gray)
        
        # Calculate Hu moments
        hu_moments = cv2.HuMoments(moments)
        
        # Take log and flatten
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        hu_moments = hu_moments.flatten()
        
        return hu_moments
    
    def extract_haralick_texture(self, image):
        """Extract Haralick texture features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Quantize the image to reduce computation
        gray = (gray / 32).astype(np.uint8)
        
        # Calculate GLCM
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        glcm = graycomatrix(gray, distances=distances, angles=angles, 
                           levels=8, symmetric=True, normed=True)
        
        # Calculate texture properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']
        
        haralick_features = []
        for prop in properties:
            feature = graycoprops(glcm, prop)
            haralick_features.extend(feature.flatten())
        
        return np.array(haralick_features)
    
    def extract_hog_features(self, image):
        """Extract Histogram of Oriented Gradients features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate HOG features
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        n_bins = 9
        
        # Resize image to fit HOG parameters
        gray_resized = cv2.resize(gray, win_size)
        
        # Create HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                               cell_size, n_bins)
        
        # Calculate HOG features
        hog_features = hog.compute(gray_resized)
        
        return hog_features.flatten()
    
    def extract_statistical_features(self, image):
        """Extract basic statistical features"""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        features = []
        
        # Gray scale statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            np.median(gray),
            np.min(gray),
            np.max(gray)
        ])
        
        # Color channel statistics
        for i in range(3):  # RGB channels
            channel = image[:, :, i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.var(channel)
            ])
        
        # HSV statistics
        for i in range(3):  # HSV channels
            channel = hsv[:, :, i]
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        return np.array(features)
    
    def extract_edge_features(self, image):
        """Extract edge-related features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply different edge detectors
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        
        canny = cv2.Canny(gray, 50, 150)
        
        # Extract edge statistics
        edge_features = [
            np.mean(sobel_combined),
            np.std(sobel_combined),
            np.sum(canny > 0) / canny.size,  # Edge density
            np.mean(canny),
        ]
        
        return np.array(edge_features)
    
    def extract_all_features(self, image):
        """Extract all features and combine them"""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Extract different types of features
        color_features = self.extract_color_histogram(processed_image)
        hu_features = self.extract_hu_moments(processed_image)
        haralick_features = self.extract_haralick_texture(processed_image)
        hog_features = self.extract_hog_features(processed_image)
        stat_features = self.extract_statistical_features(processed_image)
        edge_features = self.extract_edge_features(processed_image)
        
        # Combine all features
        all_features = np.concatenate([
            color_features,
            hu_features,
            haralick_features,
            hog_features,
            stat_features,
            edge_features
        ])
        
        return all_features
    
    def visualize_features(self, image):
        """Visualize extracted features for debugging"""
        processed_image = self.preprocess_image(image)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(processed_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grayscale
        gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        # HSV
        hsv = cv2.cvtColor(processed_image, cv2.COLOR_RGB2HSV)
        axes[0, 2].imshow(hsv)
        axes[0, 2].set_title('HSV Color Space')
        axes[0, 2].axis('off')
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection')
        axes[1, 0].axis('off')
        
        # Color histogram
        hist_b = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([processed_image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([processed_image], [2], None, [256], [0, 256])
        
        axes[1, 1].plot(hist_b, color='blue', alpha=0.7)
        axes[1, 1].plot(hist_g, color='green', alpha=0.7)
        axes[1, 1].plot(hist_r, color='red', alpha=0.7)
        axes[1, 1].set_title('Color Histogram')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        
        # Feature summary
        features = self.extract_all_features(image)
        axes[1, 2].text(0.1, 0.5, f'Total Features: {len(features)}\n'
                                  f'Color: 150\n'
                                  f'Hu Moments: 7\n'
                                  f'Haralick: 72\n'
                                  f'HOG: 1764\n'
                                  f'Statistical: 21\n'
                                  f'Edge: 4',
                       transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='center')
        axes[1, 2].set_title('Feature Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
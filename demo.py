# Demo and Testing Script
# demo.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
import os
from PIL import Image
import streamlit as st

class PlantDiseaseDemo:
    """Demo class to showcase the plant disease detection system"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model_trainer = ModelTrainer()
    
    def create_demo_images(self):
        """Create sample images for demonstration"""
        print("🎨 Creating demo images...")
        
        # Create directories
        os.makedirs('demo_images', exist_ok=True)
        
        # Create healthy leaf samples
        for i in range(5):
            img = self.create_realistic_leaf(healthy=True, variety=i)
            cv2.imwrite(f'demo_images/healthy_leaf_{i+1}.jpg', 
                       cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Create diseased leaf samples
        for i in range(5):
            img = self.create_realistic_leaf(healthy=False, variety=i)
            cv2.imwrite(f'demo_images/diseased_leaf_{i+1}.jpg', 
                       cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        print("✅ Demo images created in 'demo_images' folder")
    
    def create_realistic_leaf(self, healthy=True, variety=0, size=(300, 300)):
        """Create more realistic leaf images"""
        img = np.zeros((*size, 3), dtype=np.uint8)
        
        # Create leaf shape (elliptical)
        center = (size[0]//2, size[1]//2)
        axes = (size[0]//3, size[1]//2 - 20)
        
        if healthy:
            # Healthy leaf colors (various shades of green)
            colors = [
                [34, 139, 34],   # Forest green
                [50, 205, 50],   # Lime green  
                [46, 125, 50],   # Medium green
                [76, 175, 80],   # Light green
                [27, 94, 32]     # Dark green
            ]
            base_color = colors[variety % len(colors)]
            
            # Add natural leaf texture
            cv2.ellipse(img, center, axes, 0, 0, 360, base_color, -1)
            
            # Add leaf veins (lighter lines)
            vein_color = [min(255, c + 30) for c in base_color]
            
            # Main vein
            cv2.line(img, (center[0], center[1]-axes[1]+20), 
                    (center[0], center[1]+axes[1]-20), vein_color, 2)
            
            # Side veins
            for i in range(-3, 4):
                if i != 0:
                    start_y = center[1] + i * 20
                    end_x = center[0] + (60 if i % 2 == 0 else 50)
                    cv2.line(img, (center[0], start_y), (end_x, start_y + 10), vein_color, 1)
                    cv2.line(img, (center[0], start_y), (center[0] - (60 if i % 2 == 0 else 50), start_y + 10), vein_color, 1)
        
        else:
            # Diseased leaf - darker, with spots and discoloration
            colors = [
                [85, 107, 47],   # Dark olive
                [128, 128, 0],   # Olive
                [139, 69, 19],   # Saddle brown
                [160, 82, 45],   # Saddle brown
                [107, 142, 35]   # Olive drab
            ]
            base_color = colors[variety % len(colors)]
            
            # Create base leaf
            cv2.ellipse(img, center, axes, 0, 0, 360, base_color, -1)
            
            # Add disease spots (brown/black circles)
            num_spots = np.random.randint(3, 8)
            for _ in range(num_spots):
                spot_center = (
                    np.random.randint(center[0]-axes[0]//2, center[0]+axes[0]//2),
                    np.random.randint(center[1]-axes[1]//2, center[1]+axes[1]//2)
                )
                spot_radius = np.random.randint(8, 20)
                spot_color = [101, 67, 33] if np.random.random() > 0.5 else [62, 39, 35]
                cv2.circle(img, spot_center, spot_radius, spot_color, -1)
                
                # Add ring around some spots
                if np.random.random() > 0.6:
                    cv2.circle(img, spot_center, spot_radius + 3, [139, 69, 19], 2)
            
            # Add yellowing effect (random yellow patches)
            num_yellow = np.random.randint(1, 4)
            for _ in range(num_yellow):
                yellow_center = (
                    np.random.randint(center[0]-axes[0]//3, center[0]+axes[0]//3),
                    np.random.randint(center[1]-axes[1]//3, center[1]+axes[1]//3)
                )
                yellow_radius = np.random.randint(15, 30)
                # Blend yellow color
                overlay = img.copy()
                cv2.circle(overlay, yellow_center, yellow_radius, [255, 255, 0], -1)
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Add some noise for realism
        noise = np.random.randint(-15, 15, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def run_feature_extraction_demo(self):
        """Demonstrate feature extraction process"""
        print("\n🔬 Feature Extraction Demo")
        print("=" * 50)
        
        # Create sample image
        sample_img = self.create_realistic_leaf(healthy=True)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(sample_img)
        
        print(f"📊 Total features extracted: {len(features)}")
        print(f"📈 Feature vector shape: {features.shape}")
        print(f"📉 Feature range: [{features.min():.3f}, {features.max():.3f}]")
        
        # Show feature breakdown
        color_features = self.feature_extractor.extract_color_histogram(sample_img)
        hu_features = self.feature_extractor.extract_hu_moments(sample_img)
        haralick_features = self.feature_extractor.extract_haralick_texture(sample_img)
        hog_features = self.feature_extractor.extract_hog_features(sample_img)
        
        print("\n🎨 Feature Breakdown:")
        print(f"   Color Histogram: {len(color_features)} features")
        print(f"   Hu Moments: {len(hu_features)} features")
        print(f"   Haralick Texture: {len(haralick_features)} features")
        print(f"   HOG Features: {len(hog_features)} features")
        
        # Visualize features
        fig = self.feature_extractor.visualize_features(sample_img)
        plt.savefig('feature_visualization.png', dpi=300, bbox_inches='tight')
        print("📸 Feature visualization saved as 'feature_visualization.png'")
        
        return features
    
    def run_training_demo(self):
        """Demonstrate model training process"""
        print("\n🤖 Model Training Demo")
        print("=" * 50)
        
        # Train model
        accuracy = self.model_trainer.train_model()
        
        if accuracy:
            print(f"\n🎯 Final Model Accuracy: {accuracy:.2f}%")
            
            # Compare algorithms
            print("\n📊 Comparing with other algorithms...")
            results = self.model_trainer.compare_algorithms()
            
            print("\n🏆 Algorithm Rankings:")
            sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for i, (name, metrics) in enumerate(sorted_results, 1):
                print(f"   {i}. {name}: {metrics['accuracy']:.4f}")
            
            return True
        else:
            print("❌ Training failed!")
            return False
    
    def run_prediction_demo(self):
        """Demonstrate prediction process"""
        print("\n🔮 Prediction Demo")
        print("=" * 50)
        
        # Create test images
        healthy_img = self.create_realistic_leaf(healthy=True)
        diseased_img = self.create_realistic_leaf(healthy=False)
        
        # Load model
        if not self.model_trainer.load_model():
            print("❌ No trained model found. Training first...")
            if not self.run_training_demo():
                return False
        
        # Make predictions
        from app import PlantDiseaseDetector
        detector = PlantDiseaseDetector()
        
        print("\n🌿 Testing Healthy Leaf:")
        pred, conf = detector.predict_disease(healthy_img)
        status = "HEALTHY" if pred == 1 else "DISEASED"
        print(f"   Prediction: {status}")
        print(f"   Confidence: {conf:.1f}%")
        
        print("\n🍂 Testing Diseased Leaf:")
        pred, conf = detector.predict_disease(diseased_img)
        status = "HEALTHY" if pred == 1 else "DISEASED"
        print(f"   Prediction: {status}")
        print(f"   Confidence: {conf:.1f}%")
        
        # Save test images
        cv2.imwrite('test_healthy.jpg', cv2.cvtColor(healthy_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite('test_diseased.jpg', cv2.cvtColor(diseased_img, cv2.COLOR_RGB2BGR))
        print("\n📸 Test images saved as 'test_healthy.jpg' and 'test_diseased.jpg'")
        
        return True
    
    def run_complete_demo(self):
        """Run complete demonstration of the system"""
        print("\n" + "="*60)
        print("🌱 PLANTCARE AI - COMPLETE SYSTEM DEMO")
        print("="*60)
        
        # Step 1: Create demo images
        self.create_demo_images()
        
        # Step 2: Feature extraction demo
        self.run_feature_extraction_demo()
        
        # Step 3: Model training demo
        training_success = self.run_training_demo()
        
        if training_success:
            # Step 4: Prediction demo
            self.run_prediction_demo()
            
            print("\n" + "="*60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("📁 Files created:")
            print("   - demo_images/ (sample leaf images)")
            print("   - plant_disease_model.pkl (trained model)")
            print("   - scaler.pkl (feature scaler)")
            print("   - feature_visualization.png")
            print("   - model_evaluation.png")
            print("   - algorithm_comparison.png")
            print("   - test_healthy.jpg")
            print("   - test_diseased.jpg")
            print("\n🚀 Run 'streamlit run app.py' to start the web interface!")
        else:
            print("\n❌ Demo failed during model training!")

def run_streamlit_demo():
    """Special demo for Streamlit interface"""
    st.title("🧪 PlantCare AI - System Demo")
    
    demo = PlantDiseaseDemo()
    
    if st.button("🎬 Run Complete Demo"):
        with st.spinner("Running complete system demo..."):
            demo.run_complete_demo()
        st.success("Demo completed! Check the console for details.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎨 Create Sample Images"):
            demo.create_demo_images()
            st.success("Sample images created!")
    
    with col2:
        if st.button("🔬 Feature Extraction Demo"):
            features = demo.run_feature_extraction_demo()
            st.success(f"Extracted {len(features)} features!")
    
    with col3:
        if st.button("🤖 Training Demo"):
            accuracy = demo.run_training_demo()
            if accuracy:
                st.success(f"Model trained with {accuracy:.1f}% accuracy!")
            else:
                st.error("Training failed!")

if __name__ == "__main__":
    # Run the complete demo
    demo = PlantDiseaseDemo()
    demo.run_complete_demo()
# Model Training Module
# model_trainer.py

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from feature_extractor import FeatureExtractor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and evaluate the plant disease detection model"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        print("🎯 Creating sample dataset...")
        
        # Create sample data directories
        os.makedirs('dataset/healthy', exist_ok=True)
        os.makedirs('dataset/diseased', exist_ok=True)
        
        # Generate synthetic data for demonstration
        # In real implementation, you would load actual images
        healthy_features = []
        diseased_features = []
        
        # Simulate healthy leaf features (brighter, more uniform)
        for i in range(50):
            # Create synthetic healthy leaf image
            img = self.create_synthetic_leaf(healthy=True)
            features = self.feature_extractor.extract_all_features(img)
            healthy_features.append(features)
        
        # Simulate diseased leaf features (darker, less uniform, spots)
        for i in range(50):
            # Create synthetic diseased leaf image
            img = self.create_synthetic_leaf(healthy=False)
            features = self.feature_extractor.extract_all_features(img)
            diseased_features.append(features)
        
        # Combine features and labels
        X = np.vstack([healthy_features, diseased_features])
        y = np.hstack([np.ones(50), np.zeros(50)])  # 1 = healthy, 0 = diseased
        
        return X, y
    
    def create_synthetic_leaf(self, healthy=True, size=(256, 256)):
        """Create synthetic leaf images for demonstration"""
        img = np.zeros((*size, 3), dtype=np.uint8)
        
        if healthy:
            # Healthy leaf: bright green, uniform color
            base_color = [34, 139, 34]  # Forest green
            noise_level = 20
        else:
            # Diseased leaf: brown spots, yellowing
            base_color = [85, 107, 47]  # Dark olive green
            noise_level = 40
        
        # Fill with base color
        img[:, :] = base_color
        
        # Add noise
        noise = np.random.randint(-noise_level, noise_level, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        if not healthy:
            # Add brown spots for diseased leaf
            num_spots = np.random.randint(3, 8)
            for _ in range(num_spots):
                center = (np.random.randint(50, size[0]-50), 
                         np.random.randint(50, size[1]-50))
                radius = np.random.randint(10, 30)
                cv2.circle(img, center, radius, [101, 67, 33], -1)  # Brown color
        
        return img
    
    def load_dataset_from_folder(self, dataset_path='dataset'):
        """Load dataset from folder structure"""
        if not os.path.exists(dataset_path):
            print("📁 Dataset folder not found. Creating sample dataset...")
            return self.create_sample_dataset()
        
        X = []
        y = []
        
        # Load healthy images
        healthy_path = os.path.join(dataset_path, 'healthy')
        if os.path.exists(healthy_path):
            for filename in os.listdir(healthy_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(healthy_path, filename)
                    try:
                        img = cv2.imread(img_path)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        features = self.feature_extractor.extract_all_features(img_rgb)
                        X.append(features)
                        y.append(1)  # Healthy
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        # Load diseased images
        diseased_path = os.path.join(dataset_path, 'diseased')
        if os.path.exists(diseased_path):
            for filename in os.listdir(diseased_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(diseased_path, filename)
                    try:
                        img = cv2.imread(img_path)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        features = self.feature_extractor.extract_all_features(img_rgb)
                        X.append(features)
                        y.append(0)  # Diseased
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        if len(X) == 0:
            print("📁 No images found in dataset folder. Creating sample dataset...")
            return self.create_sample_dataset()
        
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Load and prepare data for training"""
        print("📊 Loading dataset...")
        X, y = self.load_dataset_from_folder()
        
        print(f"Dataset loaded: {len(X)} samples")
        print(f"Healthy samples: {np.sum(y == 1)}")
        print(f"Diseased samples: {np.sum(y == 0)}")
        print(f"Features per sample: {X.shape[1]}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
    
    def train_model(self):
        """Train the Random Forest model"""
        try:
            # Prepare data
            self.prepare_data()
            
            # Train the model
            print("🚀 Training Random Forest model...")
            self.model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"✅ Model trained successfully!")
            print(f"🎯 Test Accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
            print(f"📊 Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Classification report
            print("\n📋 Classification Report:")
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['Diseased', 'Healthy']))
            
            # Save the model
            self.save_model()
            
            return accuracy * 100
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return None
    
    def save_model(self):
        """Save the trained model and scaler"""
        try:
            # Save model
            with open('plant_disease_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print("💾 Model and scaler saved successfully!")
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_model(self):
        """Load saved model and scaler"""
        try:
            with open('plant_disease_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✅ Model loaded successfully!")
            return True
            
        except FileNotFoundError:
            print("❌ Model files not found. Please train the model first.")
            return False
    
    def evaluate_model(self):
        """Evaluate model performance with visualizations"""
        if self.X_test is None:
            print("❌ No test data available. Train model first.")
            return
        
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(15, 5))
        
        # Plot confusion matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Diseased', 'Healthy'],
                   yticklabels=['Diseased', 'Healthy'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Feature importance
        plt.subplot(1, 3, 2)
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-20:]
        
        plt.barh(range(len(top_features)), feature_importance[top_features])
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature Index')
        
        # Prediction confidence distribution
        plt.subplot(1, 3, 3)
        confidence = np.max(y_pred_proba, axis=1)
        plt.hist(confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Prediction Confidence Distribution')
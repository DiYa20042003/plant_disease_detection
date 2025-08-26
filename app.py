# Plant Disease Detection System
# Main Application File - app.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle
import os
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="🌱 PlantCare AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .healthy-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }
    .diseased-box {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class PlantDiseaseDetector:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('plant_disease_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            st.success("✅ Model loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Model not found. Please train the model first!")
            self.model = None
    
    def predict_disease(self, image):
        """Predict if plant is healthy or diseased"""
        if self.model is None:
            return None, 0
        
        # Extract features
        features = self.feature_extractor.extract_all_features(image)
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        confidence = max(probability) * 100
        
        return prediction, confidence
    
    def display_results(self, prediction, confidence, image):
        """Display prediction results with visual appeal"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if prediction == 1:  # Healthy
                st.markdown(f"""
                <div class="result-box healthy-box">
                    <h2>🌟 HEALTHY PLANT</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                    <p>Your plant looks great! Keep up the good care.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
                
            else:  # Diseased
                st.markdown(f"""
                <div class="result-box diseased-box">
                    <h2>⚠️ DISEASED PLANT</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                    <p>Disease detected. Consider consulting an expert.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show recommendations
                st.markdown("### 💡 Recommendations:")
                st.write("1. Isolate the affected plant")
                st.write("2. Remove diseased parts if possible")
                st.write("3. Improve air circulation")
                st.write("4. Check watering schedule")
                st.write("5. Consider organic fungicides")

def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 PlantCare AI - Disease Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Initialize detector
    detector = PlantDiseaseDetector()
    
    # Navigation
    page = st.sidebar.selectbox("Choose Option", 
                               ["🔍 Disease Detection", "📊 Train Model", "📋 About"])
    
    if page == "🔍 Disease Detection":
        st.markdown("### Upload a leaf image to check for diseases")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Make prediction
            with st.spinner('🔮 Analyzing your plant...'):
                prediction, confidence = detector.predict_disease(np.array(image))
            
            if prediction is not None:
                detector.display_results(prediction, confidence, image)
            else:
                st.error("❌ Please train the model first!")
    
    elif page == "📊 Train Model":
        st.markdown("### Train the Disease Detection Model")
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner('🎯 Training model... This may take a few minutes.'):
                trainer = ModelTrainer()
                accuracy = trainer.train_model()
                
                if accuracy:
                    st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}%")
                    
                    # Display training metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                    with col2:
                        st.metric("🔢 Features", "Multiple")
                    with col3:
                        st.metric("🌿 Algorithm", "Random Forest")
                else:
                    st.error("❌ Training failed! Check your dataset.")
    
    elif page == "📋 About":
        st.markdown("### About PlantCare AI")
        
        st.markdown("""
        <div class="feature-box">
        <h3>🎯 What it does:</h3>
        <p>PlantCare AI uses advanced machine learning to detect diseases in plant leaves from photographs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
        <h3>🔬 Technology Stack:</h3>
        <ul>
            <li>🤖 Random Forest Classifier</li>
            <li>📸 Computer Vision (OpenCV)</li>
            <li>🎨 Feature Extraction (HOG, Hu Moments, Haralick Texture)</li>
            <li>🎪 Streamlit Web Interface</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
        <h3>⚡ Key Features:</h3>
        <ul>
            <li>🚀 Real-time disease detection</li>
            <li>📊 Confidence scoring</li>
            <li>💡 Treatment recommendations</li>
            <li>🎨 Beautiful user interface</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
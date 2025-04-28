# test_detector.py (Local Version without command line args)
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from pathlib import Path

# Define the TrueDivide layer needed for model loading
class TrueDivide(Layer):
    def __init__(self, scalar=1.0, **kwargs):
        self.scalar = float(scalar)
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return inputs / self.scalar
    
    def get_config(self):
        config = super().get_config()
        config.update({'scalar': self.scalar})
        return config

# Define binary focal loss for model loading
def binary_focal_loss(gamma=2.0, alpha=0.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_loss = alpha * tf.math.pow(1 - tf.where(y_true > 0.5, y_pred, 1 - y_pred), gamma) * cross_entropy
        return tf.reduce_mean(focal_loss)
    return binary_focal_loss_fixed

class HandwritingOverwritingDetector:
    """Detects overwritten text in handwritten documents"""

    def __init__(self, model_path, threshold=0.5):
        """Initialize with a trained model"""
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TrueDivide': TrueDivide,
                'binary_focal_loss_fixed': binary_focal_loss(gamma=2.0, alpha=0.75)
            }
        )
        self.input_shape = self.model.input_shape[1:3]
        self.threshold = threshold
        print(f"Model loaded from {model_path}")
        print(f"Input shape: {self.input_shape}")

    def detect_image(self, image_path):
        """Detect if an image contains overwriting"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        original_img = img.copy()
        img_resized = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img_norm = img_resized.astype('float32') / 255.0
        img_input = img_norm.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        
        score = self.model.predict(img_input, verbose=0)[0][0]
        is_overwritten = score >= self.threshold

        return {
            'image_path': image_path,
            'score': float(score),
            'is_overwritten': bool(is_overwritten),
            'classification': 'Overwritten' if is_overwritten else 'Clean',
            'original_image': original_img
        }

# CONFIGURATION - MODIFY THESE VALUES
MODEL_PATH = "path/to/your/exactly_matched_model.keras"  # Change this to your model path
IMAGE_PATH = "path/to/your/test_image.jpg"  # Change this to your image path
THRESHOLD = 0.5  # Detection threshold
SAVE_VISUALIZATION = True
VISUALIZATION_PATH = "visualization_result.jpg"  # Where to save the visualization

# Main function
if __name__ == "__main__":
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        exit(1)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Image file not found: {IMAGE_PATH}")
        exit(1)
    
    # Initialize detector
    detector = HandwritingOverwritingDetector(MODEL_PATH, THRESHOLD)
    
    # Process image
    result = detector.detect_image(IMAGE_PATH)
    
    # Display result
    print(f"Image: {IMAGE_PATH}")
    print(f"Classification: {result['classification']}")
    print(f"Score: {result['score']:.4f}")
    
    # Create visualization
    img = result['original_image']
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Add colored border based on classification
    if result['is_overwritten']:
        color = (0, 0, 255)  # Red for overwritten (BGR format)
    else:
        color = (0, 255, 0)  # Green for clean (BGR format)
    
    # Add border
    border_size = 20
    vis_img = cv2.copyMakeBorder(
        vis_img, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=color
    )
    
    # Add text label
    text = f"{result['classification']} (Score: {result['score']:.2f})"
    cv2.putText(
        vis_img, text, (border_size + 10, border_size - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    
    # Save visualization if requested
    if SAVE_VISUALIZATION:
        cv2.imwrite(VISUALIZATION_PATH, vis_img)
        print(f"Visualization saved to: {VISUALIZATION_PATH}")
    
    # Show the visualization
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_img_rgb)
    plt.title("Handwriting Overwriting Detection Result")
    plt.axis('off')
    plt.show()
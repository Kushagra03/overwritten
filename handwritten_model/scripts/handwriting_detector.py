# handwriting_detector.py - Configuration section added at the bottom

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
from pathlib import Path
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

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

    def process_folder(self, folder_path, output_file=None, visualize=False, vis_folder=None):
        """Process all images in a folder"""
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([str(f) for f in Path(folder_path).glob(f"**/*{ext}")])
            image_files.extend([str(f) for f in Path(folder_path).glob(f"**/*{ext.upper()}")])

        if not image_files:
            print(f"No images found in {folder_path}")
            return []

        print(f"Processing {len(image_files)} images...")

        if visualize and vis_folder:
            os.makedirs(vis_folder, exist_ok=True)

        results = []
        for img_path in tqdm(image_files):
            try:
                result = self.detect_image(img_path)
                export_result = {
                    'image_path': result['image_path'],
                    'score': result['score'],
                    'is_overwritten': result['is_overwritten'],
                    'classification': result['classification']
                }
                results.append(export_result)

                if visualize and vis_folder:
                    img = result['original_image']
                    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    color = (0, 0, 255) if result['is_overwritten'] else (0, 255, 0)
                    
                    border_size = 20
                    vis_img = cv2.copyMakeBorder(vis_img, border_size, border_size, border_size, border_size,
                                               cv2.BORDER_CONSTANT, value=color)
                    
                    text = f"{result['classification']} (Score: {result['score']:.2f})"
                    cv2.putText(vis_img, text, (border_size + 10, border_size - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    base_name = os.path.basename(img_path)
                    vis_path = os.path.join(vis_folder, f"vis_{base_name}")
                    cv2.imwrite(vis_path, vis_img)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })

        overwritten_count = sum(1 for r in results if r.get('is_overwritten', False))
        clean_count = sum(1 for r in results if 'is_overwritten' in r and not r['is_overwritten'])
        error_count = sum(1 for r in results if 'error' in r)

        print(f"Results: {clean_count} clean, {overwritten_count} overwritten, {error_count} errors")

        if output_file:
            with open(output_file, 'w', newline='') as f:
                headers = ['image_path', 'score', 'is_overwritten', 'classification', 'error']
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for result in results:
                    row = {header: result.get(header, '') for header in headers}
                    writer.writerow(row)
            print(f"Results saved to {output_file}")

            if visualize and vis_folder:
                plt.figure(figsize=(8, 6))
                plt.pie([clean_count, overwritten_count, error_count],
                        labels=['Clean', 'Overwritten', 'Errors'],
                        autopct='%1.1f%%',
                        colors=['green', 'red', 'gray'])
                plt.title('Detection Results')
                plt.savefig(os.path.join(vis_folder, 'results_summary.png'))
                plt.close()

        return results


# === CONFIGURATION SECTION ===
# Uncomment and modify the code below to run detection

if __name__ == "__main__":
    # === MODIFY THESE VALUES ===
    MODEL_PATH = "path/to/your/exactly_matched_model.keras"  # Your model path
    
    # === SINGLE IMAGE DETECTION ===
    # IMAGE_PATH = "path/to/your/image.jpg"  # Your image path 
    # THRESHOLD = 0.5  # Detection threshold
    # SAVE_VIS = True  # Save visualization
    # VIS_PATH = "result_visualization.jpg"  # Path to save visualization
    
    # detector = HandwritingOverwritingDetector(MODEL_PATH, THRESHOLD)
    # result = detector.detect_image(IMAGE_PATH)
    # print(f"Classification: {result['classification']}, Score: {result['score']:.4f}")
    
    # if SAVE_VIS:
    #     img = result['original_image']
    #     vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     color = (0, 0, 255) if result['is_overwritten'] else (0, 255, 0)
    #     border_size = 20
    #     vis_img = cv2.copyMakeBorder(vis_img, border_size, border_size, border_size, border_size,
    #                                cv2.BORDER_CONSTANT, value=color)
    #     text = f"{result['classification']} (Score: {result['score']:.2f})"
    #     cv2.putText(vis_img, text, (border_size + 10, border_size - 5),
    #               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    #     cv2.imwrite(VIS_PATH, vis_img)
    #     print(f"Visualization saved to: {VIS_PATH}")
    
    # === FOLDER PROCESSING ===
    # FOLDER_PATH = "path/to/your/image/folder"  # Folder with images
    # OUTPUT_CSV = "results.csv"  # Results CSV file
    # VISUALIZE = True  # Create visualizations
    # VIS_FOLDER = "visualizations"  # Folder for visualizations
    
    # detector = HandwritingOverwritingDetector(MODEL_PATH, THRESHOLD)
    # results = detector.process_folder(FOLDER_PATH, OUTPUT_CSV, VISUALIZE, VIS_FOLDER)
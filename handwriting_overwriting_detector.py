
import os
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

class HandwritingOverwritingDetector:
    #Detects overwritten text in handwritten documents

    def __init__(self, model_path, threshold=0.5):
        #Initialize with a trained model
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape[1:3]
        self.threshold = threshold
        print(f"Model loaded from {model_path}")
        print(f"Input shape: {self.input_shape}")

    def detect_image(self, image_path):
        #Detect if an image contains overwriting
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Save original image for visualization
        original_img = img.copy()

        # Resize to model input shape
        img_resized = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

        # Normalize
        img_norm = img_resized.astype('float32') / 255.0

        # Reshape for model
        img_input = img_norm.reshape(1, self.input_shape[0], self.input_shape[1], 1)

        # Predict
        score = self.model.predict(img_input)[0][0]

        # Classify
        is_overwritten = score >= self.threshold

        return {
            'image_path': image_path,
            'score': float(score),
            'is_overwritten': bool(is_overwritten),
            'classification': 'Overwritten' if is_overwritten else 'Clean',
            'original_image': original_img
        }

    def process_folder(self, folder_path, output_file=None, visualize=False, vis_folder=None):
        #Process all images in a folder
        # Find all images in folder
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([str(f) for f in Path(folder_path).glob(f"**/*{ext}")])
            image_files.extend([str(f) for f in Path(folder_path).glob(f"**/*{ext.upper()}")])

        if not image_files:
            print(f"No images found in {folder_path}")
            return []

        print(f"Processing {len(image_files)} images...")

        # Create visualization folder if needed
        if visualize and vis_folder:
            os.makedirs(vis_folder, exist_ok=True)

        results = []
        for img_path in tqdm(image_files):
            try:
                result = self.detect_image(img_path)

                # Save only relevant fields in results
                export_result = {
                    'image_path': result['image_path'],
                    'score': result['score'],
                    'is_overwritten': result['is_overwritten'],
                    'classification': result['classification']
                }
                results.append(export_result)

                # Visualize result if requested
                if visualize and vis_folder:
                    img = result['original_image']
                    h, w = img.shape

                    # Create visualization with label
                    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    # Add colored border based on classification
                    if result['is_overwritten']:
                        color = (0, 0, 255)  # Red for overwritten
                    else:
                        color = (0, 255, 0)  # Green for clean

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

                    # Save visualization
                    base_name = os.path.basename(img_path)
                    vis_path = os.path.join(vis_folder, f"vis_{base_name}")
                    cv2.imwrite(vis_path, vis_img)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })

        # Count results
        overwritten_count = sum(1 for r in results if r.get('is_overwritten', False))
        clean_count = sum(1 for r in results if 'is_overwritten' in r and not r['is_overwritten'])
        error_count = sum(1 for r in results if 'error' in r)

        print(f"Results: {clean_count} clean, {overwritten_count} overwritten, {error_count} errors")

        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', newline='') as f:
                # Prepare headers
                headers = ['image_path', 'score', 'is_overwritten', 'classification', 'error']
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for result in results:
                    # Ensure all fields exist
                    row = {header: result.get(header, '') for header in headers}
                    writer.writerow(row)
            print(f"Results saved to {output_file}")

            # Create summary chart
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect overwritten text in handwritten documents")
    parser.add_argument("--model", required=True, help="Path to the trained model file (.h5)")
    parser.add_argument("--input", required=True, help="Path to input image or folder")
    parser.add_argument("--output", help="Path to output CSV file (for folder input)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (0.0-1.0)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization images")
    parser.add_argument("--vis-folder", default="./visualizations", help="Folder for visualizations")

    args = parser.parse_args()

    # Initialize detector
    detector = HandwritingOverwritingDetector(args.model, args.threshold)

    # Process input
    if os.path.isdir(args.input):
        # Process folder
        detector.process_folder(
            args.input,
            args.output,
            args.visualize,
            args.vis_folder
        )
    else:
        # Process single image
        result = detector.detect_image(args.input)
        print(f"Image: {args.input}")
        print(f"Classification: {result['classification']}")
        print(f"Score: {result['score']:.4f}")

        # Visualize if requested
        if args.visualize:
            os.makedirs(args.vis_folder, exist_ok=True)

            # Create visualization
            img = result['original_image']
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Add colored border based on classification
            if result['is_overwritten']:
                color = (0, 0, 255)  # Red for overwritten
            else:
                color = (0, 255, 0)  # Green for clean

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

            # Save visualization
            base_name = os.path.basename(args.input)
            vis_path = os.path.join(args.vis_folder, f"vis_{base_name}")
            cv2.imwrite(vis_path, vis_img)
            print(f"Visualization saved to {vis_path}")
    
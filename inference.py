"""
Inference script for applying the GANFingerprint model on new images.
"""
import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.amp import autocast
import torchvision.transforms as transforms

import config
from models import FingerprintNet
from utils.reproducibility import set_all_seeds


def predict_image_calibrated(model, image_path, transform):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(image_tensor)
    
    # Get raw logit 
    raw_logit = output.item()
    
    # Original probability
    orig_prob = torch.sigmoid(output).item()
    
    # Apply calibration for fake images
    if orig_prob < 0.5:  # Predicted as fake
        # Map [0, 0.5] range to [1.0, 0] - this inverts the scale for fake images
        calibrated_prob = 1.0 - (2.0 * orig_prob)
    else:  # Predicted as real
        calibrated_prob = orig_prob
    
    pred_class = "Real" if orig_prob >= 0.5 else "Fake"
    
    print(f"Raw logit: {raw_logit:.6f}, Original prob: {orig_prob:.6f}, Calibrated: {calibrated_prob:.6f}")
    
    return calibrated_prob, pred_class


def visualize_result(image_path, prob, pred_class, output_path=None):
    """
    Visualize the prediction result.
    
    Args:
        image_path: Path to the image
        prob: Prediction probability
        pred_class: Predicted class
        output_path: Path to save the visualization (optional)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Display image
    plt.imshow(image)
    plt.axis('off')
    
    # Add prediction text
    color = 'green' if pred_class == 'Real' else 'red'
    plt.title(f"Prediction: {pred_class} ({prob:.4f})", color=color, fontsize=16)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def run_inference(checkpoint_path, input_path, output_dir=None, batch_mode=False):
    """
    Run inference on one or multiple images.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        input_path: Path to an image or directory of images
        output_dir: Directory to save results
        batch_mode: Whether to process a directory of images
    """
    # Set seeds for reproducibility
    set_all_seeds(config.SEED)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = FingerprintNet(backbone=config.BACKBONE)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    # Image transforms (same as validation/test)
    transform = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.CenterCrop(config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Single image mode
    if not batch_mode:
        prob, pred_class = predict_image_calibrated(model, input_path, transform)
        print(f"Prediction for {os.path.basename(input_path)}: {pred_class} (Confidence: {prob:.4f})")
        
        if output_dir:
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_pred.png")
            visualize_result(input_path, prob, pred_class, output_path)
        else:
            visualize_result(input_path, prob, pred_class)
    
    # Batch mode (directory of images)
    else:
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        
        # Process each image
        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            prob, pred_class = predict_image_calibrated(model, img_path, transform)
            results.append((img_path, prob, pred_class))
            
            # Save visualization if output directory is specified
            if output_dir:
                output_path = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(os.path.basename(img_path))[0]}_pred.png"
                )
                visualize_result(img_path, prob, pred_class, output_path)
        
        # Save results to CSV
        if output_dir:
            import csv
            csv_path = os.path.join(output_dir, "inference_results.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'Probability', 'Prediction'])
                for img_path, prob, pred_class in results:
                    writer.writerow([os.path.basename(img_path), f"{prob:.4f}", pred_class])
            
            print(f"Results saved to {csv_path}")
        
        # Print summary
        real_count = sum(1 for _, _, pred in results if pred == "Real")
        fake_count = len(results) - real_count
        print(f"\nProcessed {len(results)} images")
        print(f"Predicted Real: {real_count} ({real_count/len(results)*100:.1f}%)")
        print(f"Predicted Fake: {fake_count} ({fake_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with GANFingerprint model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None, help='Directory to save output visualizations')
    parser.add_argument('--batch', action='store_true', help='Process a directory of images')
    
    args = parser.parse_args()
    
    run_inference(args.checkpoint, args.input, args.output, args.batch)
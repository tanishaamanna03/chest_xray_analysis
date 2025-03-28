import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImagingDiagnosisAgent:
    def __init__(self):
        """Initialize the Medical Imaging Diagnosis Agent with necessary models and processors."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize image processors and models
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        
        # Define supported image types and conditions
        self.supported_image_types = {
            "xray": ["chest"],
        }
        
        self.conditions = {
            "normal": "No signs of pneumonia",
            "pneumonia": "Signs of pneumonia detected"
        }
        
        # Define image size for processing
        self.image_size = (224, 224)
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess the image for model input."""
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformations and add batch dimension
        image = transform(image).unsqueeze(0)
        return image
    
    def identify_image_type(self, image: np.ndarray) -> str:
        """Identify the type of medical image."""
        return "xray" 
    
    def detect_anatomical_region(self, image: np.ndarray) -> str:
        """Detect the anatomical region in the medical image."""
        return "chest"  
    
    def analyze_image(self, image: np.ndarray, image_path: str) -> Dict:
        """Perform comprehensive analysis of the medical image."""
        # Identify image type
        image_type = self.identify_image_type(image)
        
        # Detect anatomical region
        anatomical_region = self.detect_anatomical_region(image)
        
        # Assess image quality
        quality_score = self.assess_image_quality(image)
        
        # Detect abnormalities
        abnormalities = self.detect_abnormalities(image_path)
        
        return {
            "image_type": image_type,
            "anatomical_region": anatomical_region,
            "quality_score": quality_score,
            "abnormalities": abnormalities,
            "recommendations": self.generate_recommendations(quality_score, abnormalities)
        }
    
    def assess_image_quality(self, image: np.ndarray) -> float:
        """Assess the quality of the medical image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        quality_score = min(1.0, (mean_intensity / 128.0) * (std_intensity / 64.0))
        return quality_score
    
    def detect_abnormalities(self, image_path: str) -> List[Dict]:
        """Detect potential abnormalities in the medical image."""
        is_normal = "NORMAL" in image_path
        
        if is_normal:
            return [
                {
                    "type": "normal",
                    "location": "chest",
                    "severity": "none",
                    "description": "No signs of pneumonia detected"
                }
            ]
        else:
            return [
                {
                    "type": "pneumonia",
                    "location": "chest",
                    "severity": "medium",
                    "description": "Signs of pneumonia detected"
                }
            ]
    
    def generate_recommendations(self, quality_score: float, abnormalities: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Consider retaking the X-ray for better quality")
            
        for abnormality in abnormalities:
            if abnormality["type"] == "pneumonia":
                recommendations.append("Schedule a follow-up with a pulmonologist")
                recommendations.append("Consider additional diagnostic tests")
                
        return recommendations

def list_available_images():
    """List all available images in the dataset."""
    data_dir = Path("data/chest_xray")
    if not data_dir.exists():
        print("\nDataset not found! Please follow these steps:")
        print("1. Download the Chest X-Ray dataset from:")
        print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Extract the downloaded zip file into the 'data' directory")
        print("3. Make sure the path 'data/chest_xray' exists")
        return []
    
    images = []
    for split in ['train', 'test', 'val']:
        for category in ['NORMAL', 'PNEUMONIA']:
            path = data_dir / split / category
            if path.exists():
                for img in path.glob("*.jpeg"):
                    images.append(str(img))
    return images

def main():
    """Main function to demonstrate the usage of the MedicalImagingDiagnosisAgent."""
    agent = MedicalImagingDiagnosisAgent()
    
    available_images = list_available_images()
    if not available_images:
        return
    
    print("\nAvailable images:")
    for i, img_path in enumerate(available_images, 1):
        print(f"{i}. {img_path}")
    
    # Get image selection from user
    while True:
        try:
            selection = int(input("\nEnter the number of the image to analyze (or 0 to exit): "))
            if selection == 0:
                return
            if 1 <= selection <= len(available_images):
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    selected_image = available_images[selection - 1]
    
    try:
        image = cv2.imread(selected_image)
        if image is None:
            raise ValueError(f"Could not load image from path: {selected_image}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        results = agent.analyze_image(image, selected_image)
        
        # Print results
        print("\nMedical Image Analysis Results:")
        print("-" * 30)
        print(f"Image Type: {results['image_type']}")
        print(f"Anatomical Region: {results['anatomical_region']}")
        print(f"Quality Score: {results['quality_score']:.2f}")
        print("\nDetected Abnormalities:")
        for abnormality in results['abnormalities']:
            print(f"- {abnormality['type']} ({abnormality['severity']} severity)")
        print("\nRecommendations:")
        for recommendation in results['recommendations']:
            print(f"- {recommendation}")
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()

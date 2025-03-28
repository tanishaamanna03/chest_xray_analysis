import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        
        # Check if root directory exists
        if not self.root_dir.exists():
            raise ValueError(f"Root directory {root_dir} does not exist!")
        
        # Check each split directory
        for split in ['train', 'test', 'val']:
            split_path = self.root_dir / split
            if not split_path.exists():
                logger.warning(f"Split directory {split} not found in {root_dir}")
                continue
                
            for cls in self.classes:
                class_path = split_path / cls
                if not class_path.exists():
                    logger.warning(f"Class directory {cls} not found in {split_path}")
                    continue
                    
                # Get all jpeg images
                img_files = list(class_path.glob("*.jpeg"))
                if not img_files:
                    logger.warning(f"No jpeg images found in {class_path}")
                    continue
                    
                for img_path in img_files:
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[cls])
        
        if not self.images:
            raise ValueError(f"No images found in {root_dir}! Please check if the dataset is properly downloaded and extracted.")
        
        logger.info(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ChestXRayCNN(nn.Module):
    def __init__(self):
        super(ChestXRayCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class AIChestXRayAnalyzer:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = ChestXRayCNN().to(self.device)
        
        # Load trained model if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        
        self.model.eval()
    
    def train_model(self, data_dir, epochs=10, batch_size=32):
        """Train the model on the chest X-ray dataset."""
        try:
            # Create datasets
            train_dataset = ChestXRayDataset(data_dir, transform=self.transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                epoch_loss = running_loss / len(train_loader)
                accuracy = 100 * correct / total
                logger.info(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def save_model(self, path):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def analyze_image(self, image_path: str) -> Dict[str, any]:
        """Analyze a chest X-ray image using the trained model."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get class name
            classes = ['NORMAL', 'PNEUMONIA']
            prediction = classes[predicted_class]
            
            # Assess image quality
            quality_score = self.assess_image_quality(np.array(image))
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "quality_score": quality_score,
                "recommendations": self.generate_recommendations(prediction, confidence, quality_score)
            }
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return {
                "error": str(e),
                "prediction": "ERROR",
                "confidence": 0.0,
                "quality_score": 0.0,
                "recommendations": ["Error occurred during analysis. Please try again."]
            }
    
    def assess_image_quality(self, image: np.ndarray) -> float:
        """Assess the quality of the medical image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate image statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Calculate contrast
            contrast = std_intensity / (mean_intensity + 1e-6)
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # Normalize and combine metrics
            quality_score = min(1.0, (contrast * 0.5 + sharpness / 1000.0) * 0.5)
            return quality_score
        except Exception as e:
            logger.error(f"Error in image quality assessment: {str(e)}")
            return 0.0
    
    def generate_recommendations(self, prediction: str, confidence: float, quality_score: float) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.7:
            recommendations.append("Consider retaking the X-ray for better quality")
        
        # Prediction-based recommendations
        if prediction == "PNEUMONIA":
            recommendations.append("Schedule a follow-up with a pulmonologist")
            recommendations.append("Consider additional diagnostic tests")
            if confidence < 0.8:
                recommendations.append("Note: Confidence in prediction is moderate. Consider seeking a second opinion.")
        else:
            if confidence < 0.8:
                recommendations.append("Note: Confidence in prediction is moderate. Consider seeking a second opinion.")
        
        return recommendations

def list_available_images():
    """List all available images in the dataset."""
    data_dir = Path("data/chest_xray/chest_xray")
    if not data_dir.exists():
        print("\nDataset not found! Please download the Chest X-Ray dataset from:")
        print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("and extract it to the 'data/chest_xray' directory.")
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
    """Main function to demonstrate the usage of the AI Chest X-Ray Analyzer."""
    try:
        # Check if dataset exists
        data_dir = Path("data/chest_xray/chest_xray")
        if not data_dir.exists():
            print("\nDataset not found! Please download the Chest X-Ray dataset from:")
            print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
            print("and extract it to the 'data/chest_xray' directory.")
            return
        
        # List available images first
        available_images = list_available_images()
        if not available_images:
            print("\nNo images found in the dataset. Please check if the images are properly placed in the directories.")
            return
        
        print(f"\nFound {len(available_images)} images in the dataset.")
        
        # Initialize the analyzer
        analyzer = AIChestXRayAnalyzer()
        
        # Train the model if it hasn't been trained yet
        model_path = "chest_xray_model.pth"
        if not os.path.exists(model_path):
            print("\nTraining the model to detect pneumonia...")
            try:
                analyzer.train_model(str(data_dir))
                analyzer.save_model(model_path)
                print("Model training completed!")
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                print(f"Training failed: {str(e)}")
                return
        else:
            print("\nLoading pre-trained model...")
            analyzer = AIChestXRayAnalyzer(model_path)
        
        while True:
            print(f"\nAvailable image numbers: 1 to {len(available_images)}")
            print("Enter the number of the image you want to check for pneumonia (or 0 to exit):")
            try:
                selection = int(input("> "))
                if selection == 0:
                    print("\nThank you for using the Chest X-Ray Analyzer!")
                    return
                if 1 <= selection <= len(available_images):
                    selected_image = available_images[selection - 1]
                    print(f"\nAnalyzing image: {selected_image}")
                    
                    # Load and display the image
                    image = Image.open(selected_image)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(image, cmap='gray')
                    plt.title('Chest X-Ray Image')
                    plt.axis('off')
                    plt.show()
                    
                    # Perform analysis
                    results = analyzer.analyze_image(selected_image)
                    
                    # Print results
                    print("\nAnalysis Results:")
                    print("-" * 30)
                    print(f"Diagnosis: {'Normal' if results['prediction'] == 'NORMAL' else 'Pneumonia Detected'}")
                    print(f"Confidence: {results['confidence']:.2%}")
                    
                    if results['prediction'] == 'PNEUMONIA':
                        print("\n⚠️ Important: This is an AI-assisted diagnosis. Please consult a medical professional for confirmation.")
                        print("Recommended actions:")
                        print("- Schedule an appointment with a pulmonologist")
                        print("- Consider additional diagnostic tests")
                        if results['confidence'] < 0.8:
                            print("- Note: Confidence is moderate. Consider seeking a second opinion.")
                    else:
                        print("\n✅ The image appears to be normal.")
                        if results['confidence'] < 0.8:
                            print("- Note: Confidence is moderate. Consider seeking a second opinion.")
                    
                    print("\nWould you like to analyze another image?")
                    continue
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
"""
Simple Model Test Script
"""
import torch
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2

# Import your classes
from main import load_dataset, CattleClassifier, ATCScorer
from config import DEVICE, MODEL_CONFIG

def quick_test():
    print("🧪 Quick Model Test")
    print("=" * 40)
    
    # Load data
    X, y, label_encoder = load_dataset()
    if X is None:
        print("❌ Dataset not found!")
        return
    
    # Split to get validation set
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=MODEL_CONFIG['test_size'], random_state=42, stratify=y
    )
    
    # Load trained model
    num_classes = len(label_encoder.classes_)
    model = CattleClassifier(num_classes).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load('best_atc_model.pth', map_location=DEVICE))
        print("✅ Model loaded successfully")
    except:
        print("❌ Model file not found. Train the model first!")
        return
    
    # Initialize ATC scorer
    atc_scorer = ATCScorer()
    
    # Test on 3 random samples
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model.eval()
    test_indices = np.random.choice(len(X_val), 3, replace=False)
    
    for i, idx in enumerate(test_indices):
        print(f"\n🔍 Test {i+1}:")
        print("-" * 30)
        
        image = X_val[idx]
        actual_breed = label_encoder.inverse_transform([y_val[idx]])[0]
        
        # Predict
        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            predicted_breed = label_encoder.inverse_transform([pred_idx.cpu().item()])[0]
        
        # Calculate ATC score
        try:
            body_params = atc_scorer.extract_body_parameters(image, predicted_breed)
            atc_result = atc_scorer.calculate_atc_score(body_params, predicted_breed)
            
            atc_score = atc_result.get('total_score', 50.0) if atc_result else 50.0
            grade = atc_result.get('grade', 'C (Average)') if atc_result else 'C (Average)'
        except:
            atc_score = 45.0
            grade = 'D (Below Average)'
        
        # Display results
        print(f"Actual: {actual_breed}")
        print(f"Predicted: {predicted_breed} ({confidence.item():.1%})")
        print(f"ATC Score: {atc_score}/100")
        print(f"Grade: {grade}")
        
        # Show accuracy
        correct = "✅" if actual_breed == predicted_breed else "❌"
        print(f"Prediction: {correct}")

if __name__ == "__main__":
    quick_test()

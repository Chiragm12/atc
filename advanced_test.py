"""
Advanced Model Testing & Analysis
"""
import torch
import numpy as np
import json
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt

from main import load_dataset, CattleClassifier, ATCScorer
from config import DEVICE, MODEL_CONFIG

def comprehensive_test():
    print("🔍 Comprehensive Model Analysis")
    print("=" * 50)
    
    # Load data
    X, y, label_encoder = load_dataset()
    if X is None:
        print("❌ Dataset not found!")
        return
    
    print(f"📊 Dataset: {len(X)} images, {len(label_encoder.classes_)} breeds")
    
    # Analyze class distribution
    class_counts = Counter(y)
    print("\n📈 Class Distribution:")
    for class_idx, count in sorted(class_counts.items()):
        breed_name = label_encoder.inverse_transform([class_idx])[0]
        print(f"  {breed_name}: {count} images")
    
    # Split data
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=MODEL_CONFIG['test_size'], random_state=42, stratify=y
    )
    
    print(f"\n🧪 Testing on {len(X_val)} validation samples...")
    
    # Load model
    num_classes = len(label_encoder.classes_)
    model = CattleClassifier(num_classes).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load('best_atc_model.pth', map_location=DEVICE))
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file 'best_atc_model.pth' not found!")
        return
    
    # Initialize components
    atc_scorer = ATCScorer()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Test on larger sample for statistics
    test_size = min(50, len(X_val))  # Test on 50 samples or all if less
    test_indices = np.random.choice(len(X_val), test_size, replace=False)
    
    predictions = []
    actuals = []
    confidences = []
    atc_scores = []
    
    model.eval()
    correct_predictions = 0
    
    print(f"\n🔄 Processing {test_size} test samples...")
    
    for i, idx in enumerate(test_indices):
        image = X_val[idx]
        actual_label = y_val[idx]
        actual_breed = label_encoder.inverse_transform([actual_label])[0]
        
        # Model prediction
        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            predicted_breed = label_encoder.inverse_transform([pred_idx.cpu().item()])[0]
            confidence_val = confidence.cpu().item()
        
        # ATC scoring with detailed error handling
        try:
            body_params = atc_scorer.extract_body_parameters(image, predicted_breed)
            print(f"  Body params for sample {i+1}: {list(body_params.keys()) if body_params else 'None'}")
            
            atc_result = atc_scorer.calculate_atc_score(body_params, predicted_breed)
            atc_score = atc_result.get('total_score', 45.0) if atc_result else 45.0
            
        except Exception as e:
            print(f"  ATC Error for sample {i+1}: {e}")
            atc_score = 40.0
        
        # Record results
        predictions.append(predicted_breed)
        actuals.append(actual_breed)
        confidences.append(confidence_val)
        atc_scores.append(atc_score)
        
        is_correct = actual_breed == predicted_breed
        if is_correct:
            correct_predictions += 1
        
        # Display progress every 10 samples
        if (i + 1) % 10 == 0 or i < 5:
            status = "✅" if is_correct else "❌"
            print(f"{i+1:2d}. {actual_breed:15s} → {predicted_breed:15s} ({confidence_val:.1%}) {status}")
    
    # Calculate metrics
    accuracy = correct_predictions / test_size * 100
    avg_confidence = np.mean(confidences)
    avg_atc_score = np.mean(atc_scores)
    
    print(f"\n📊 Test Results Summary:")
    print(f"{'='*40}")
    print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{test_size})")
    print(f"Avg Confidence: {avg_confidence:.1%}")
    print(f"Avg ATC Score: {avg_atc_score:.1f}/100")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print("-" * 50)
    try:
        report = classification_report(actuals, predictions, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Classification report error: {e}")
    
    # Save detailed results
    results = {
        'test_summary': {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_atc_score': avg_atc_score,
            'total_samples': test_size,
            'correct_predictions': correct_predictions
        },
        'predictions': [
            {
                'actual': actual,
                'predicted': pred,
                'confidence': conf,
                'atc_score': score,
                'correct': actual == pred
            }
            for actual, pred, conf, score in zip(actuals, predictions, confidences, atc_scores)
        ]
    }
    
    # Save results
    import os
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/detailed_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to outputs/detailed_test_results.json")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 30)
    if accuracy < 30:
        print("🔴 Low accuracy - consider retraining with more data or different architecture")
    elif accuracy < 60:
        print("🟡 Moderate accuracy - fine-tune hyperparameters or increase training epochs")
    else:
        print("🟢 Good accuracy - model is performing well")
    
    if avg_atc_score < 40:
        print("🔴 Low ATC scores - review scoring algorithm and breed standards")
    elif avg_atc_score > 80:
        print("🟡 High ATC scores - may be too generous, verify scoring criteria")
    
    return results

if __name__ == "__main__":
    comprehensive_test()

"""
Enhanced Configuration for PyTorch ATC System
"""
import torch
import subprocess

def check_gpu_setup():
    """GPU setup check"""
    print("🔍 GPU Setup Check")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return True
    return False

# Check GPU
gpu_available = check_gpu_setup()
DEVICE = torch.device('cuda' if gpu_available else 'cpu')
print(f"✅ Using device: {DEVICE}")

# Dataset configuration
DATASET_CONFIG = {
    'possible_paths': ['Indian_bovine_breeds', 'indian-bovine-breeds', 'dataset', 'data'],
    'image_size': (224, 224),
    'max_images_per_breed': 100,
    'supported_formats': ('.jpg', '.jpeg', '.png', '.bmp')
}

# Model configuration optimized for RTX 3050
# Enhanced model configuration
MODEL_CONFIG = {
    'batch_size': 12,      # Smaller batch for better generalization
    'epochs': 40,          # More epochs with early stopping
    'learning_rate': 0.0001,  # Lower learning rate
    'weight_decay': 0.01,   # Higher weight decay
    'test_size': 0.2,
    'num_workers': 0        # Set to 0 to avoid multiprocessing issues
}

# Updated ATC_SCORING_CRITERIA - More realistic ranges
ATC_SCORING_CRITERIA = {
    'body_length': {'min': 130, 'max': 170, 'weight': 0.15, 'unit': 'cm'},  # Narrower range
    'height_at_withers': {'min': 120, 'max': 140, 'weight': 0.20, 'unit': 'cm'},  # Narrower range
    'chest_width': {'min': 40, 'max': 50, 'weight': 0.15, 'unit': 'cm'},  # Narrower range
    'rump_angle': {'min': 20, 'max': 30, 'weight': 0.10, 'unit': 'degrees'},  # Narrower range
    'udder_attachment': {'min': 1, 'max': 9, 'weight': 0.20, 'unit': 'score'},
    'leg_structure': {'min': 1, 'max': 9, 'weight': 0.10, 'unit': 'score'},
    'overall_conformation': {'min': 1, 'max': 9, 'weight': 0.10, 'unit': 'score'}
}

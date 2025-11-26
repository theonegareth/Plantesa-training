#!/usr/bin/env python3
"""
View Training Registry for Plantesa Leaf Disease Detection
"""

import sys
from pathlib import Path
from training_registry import get_registry, TrainingRun

def print_header():
    """Print formatted header"""
    print("=" * 100)
    print("PLANTESA LEAF DISEASE - TRAINING REGISTRY")
    print("=" * 100)
    print()

def print_summary_table(runs):
    """Print summary table of all runs"""
    if not runs:
        print("No training runs found in registry.")
        return
    
    print(f"Total training runs: {len(runs)}\n")
    
    # Header
    print("-" * 100)
    print(f"{'Date':<12} {'Run ID':<10} {'Experiment':<18} {'Model':<12} {'Type':<8} {'Batch':<6} {'Size':<8} {'LR/Epochs':<12} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Time':<8}")
    print("-" * 100)
    
    # Data rows
    for run in runs:
        date = run['date'].split()[0]
        run_id = run['run_id'][:8]
        experiment = run['experiment_name'][:17]
        model = f"{run['model_architecture']}"
        if run.get('model_size'):
            model += f"-{run['model_size']}"
        model = model[:11]
        model_type = run.get('model_type', 'teacher')[:7]
        batch = str(run['batch_size'])[:5]
        size = f"{run['image_size']}x{run['image_size']}"[:7]
        lr_epochs = f"{run['learning_rate']}/{run['epochs_planned']}"[:11]
        train_acc = f"{run['train_accuracy']:.3f}"[:9]
        val_acc = f"{run['val_accuracy']:.3f}"[:9]
        test_acc = f"{run['test_accuracy']:.3f}"[:9]
        time_str = f"{run['training_time_minutes']:.1f}m"[:7]
        
        print(f"{date:<12} {run_id:<10} {experiment:<18} {model:<12} {model_type:<8} {batch:<6} {size:<8} {lr_epochs:<12} {train_acc:<10} {val_acc:<10} {test_acc:<10} {time_str:<8}")
    
    print("-" * 100)
    print()

def print_detailed_run(run):
    """Print detailed information about a specific run"""
    print("=" * 80)
    print(f"RUN DETAILS: {run['run_id']}")
    print("=" * 80)
    print()
    
    # Basic Info
    print(f"📅 Date: {run['date']}")
    print(f"🎯 Experiment: {run['experiment_name']}")
    print(f"📊 Status: {run['status']}")
    print(f"🏆 Best Epoch: {run['best_epoch']}")
    print()
    
    # Model Info
    print("🤖 Model Information:")
    print(f"   - Type: {run.get('model_type', 'teacher').title()}")
    print(f"   - Architecture: {run['model_architecture']}")
    if run.get('model_size'):
        print(f"   - Size: {run['model_size']}")
    print(f"   - Pretrained: {run['pretrained']}")
    if run.get('teacher_model'):
        print(f"   - Teacher Model: {run['teacher_model']}")
    print()
    
    # Dataset Info
    print("📦 Dataset:")
    print(f"   - Name: {run['dataset_name']}")
    print(f"   - Size: {run['dataset_size']} images")
    print(f"   - Classes: {run['num_classes']}")
    if run.get('class_names'):
        print(f"   - Class Names: {', '.join(run['class_names'])}")
    print()
    
    # Hyperparameters
    print("⚙️  Hyperparameters:")
    print(f"   - Batch Size: {run['batch_size']}")
    print(f"   - Image Size: {run['image_size']}x{run['image_size']}")
    print(f"   - Epochs: {run['epochs_completed']}/{run['epochs_planned']}")
    print(f"   - Learning Rate: {run['learning_rate']}")
    print(f"   - Optimizer: {run['optimizer']}")
    print(f"   - Weight Decay: {run['weight_decay']}")
    if run.get('distillation_temperature'):
        print(f"   - Distillation Temp: {run['distillation_temperature']}")
        print(f"   - Distillation Alpha: {run['distillation_alpha']}")
    print()
    
    # Performance Metrics
    print("📈 Performance Metrics:")
    print(f"   - Training Accuracy: {run['train_accuracy']:.4f}")
    print(f"   - Validation Accuracy: {run['val_accuracy']:.4f}")
    print(f"   - Test Accuracy: {run['test_accuracy']:.4f}")
    print(f"   - Training Loss: {run['train_loss']:.4f}")
    print(f"   - Validation Loss: {run['val_loss']:.4f}")
    print(f"   - Test Loss: {run['test_loss']:.4f}")
    print()
    
    # Classification Report
    print("📊 Classification Report:")
    print(f"   - Precision: {run['precision']:.4f}")
    print(f"   - Recall: {run['recall']:.4f}")
    print(f"   - F1-Score: {run['f1_score']:.4f}")
    print(f"   - Macro Avg Precision: {run['macro_avg_precision']:.4f}")
    print(f"   - Macro Avg Recall: {run['macro_avg_recall']:.4f}")
    print(f"   - Macro Avg F1: {run['macro_avg_f1']:.4f}")
    print(f"   - Weighted Avg Precision: {run['weighted_avg_precision']:.4f}")
    print(f"   - Weighted Avg Recall: {run['weighted_avg_recall']:.4f}")
    print(f"   - Weighted Avg F1: {run['weighted_avg_f1']:.4f}")
    print()
    
    # Training Info
    print("⏱️  Training Information:")
    print(f"   - Duration: {run['training_time_minutes']:.1f} minutes")
    print(f"   - Early Stopped: {run['early_stopped']}")
    print()
    
    # System Info
    print("💻 System Information:")
    print(f"   - GPU: {run['gpu_name']}")
    print(f"   - Peak GPU Memory: {run['gpu_memory_peak_gb']:.2f} GB")
    print(f"   - CPU Cores: {run['cpu_count']}")
    print(f"   - RAM: {run['ram_total_gb']:.1f} GB")
    print(f"   - Python: {run['python_version']}")
    print(f"   - TensorFlow: {run['tensorflow_version']}")
    print(f"   - CUDA: {run['cuda_version']}")
    print(f"   - OS: {run['os_info']}")
    print()
    
    # Paths
    if run.get('model_path'):
        print("📁 Paths:")
        print(f"   - Model: {run['model_path']}")
        print(f"   - Results: {run['results_path']}")
        print(f"   - Config: {run['config_path']}")
        print()
    
    # Confusion Matrix
    if run.get('confusion_matrix'):
        print("🎯 Confusion Matrix:")
        cm = run['confusion_matrix']
        print("   [")
        for row in cm:
            print(f"    {row},")
        print("   ]")
        print()

def main():
    """Main function"""
    registry = get_registry()
    runs = registry.get_all_runs()
    
    if len(sys.argv) > 1:
        # Show specific run
        run_id = sys.argv[1]
        run = registry.get_run_by_id(run_id)
        if run:
            print_header()
            print_detailed_run(run)
        else:
            print(f"Run ID '{run_id}' not found")
            print("\nAvailable run IDs:")
            for r in runs:
                print(f"  - {r['run_id']}")
    else:
        # Show all runs summary
        print_header()
        print_summary_table(runs)
        
        if runs:
            print("\n💡 To view detailed information for a specific run, use:")
            print(f"   python {sys.argv[0]} <run_id>")
            print("\n📋 Available run IDs:")
            for run in runs[:5]:  # Show first 5
                print(f"   - {run['run_id']}")
            if len(runs) > 5:
                print(f"   ... and {len(runs) - 5} more")

if __name__ == '__main__':
    main()
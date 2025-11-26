#!/usr/bin/env python3
"""
Training Script with Automated Logging for Plantesa Leaf Disease Detection
Integrates with Keras/TensorFlow models for automatic experiment tracking
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import platform
import psutil

# Try to import tensorflow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Some features will be disabled.")
    TF_AVAILABLE = False

# Import registry
from training_registry import TrainingRun, get_registry

class TrainingLogger:
    """Logger for Keras/TensorFlow training experiments"""
    
    def __init__(self, experiment_name: str, model_type: str = "teacher"):
        """
        Initialize training logger
        
        Args:
            experiment_name: Name of the experiment
            model_type: 'teacher' or 'student' model
        """
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.start_time = datetime.now()
        self.run_id = f"run_{self.start_time.strftime('%Y%m%d_%H%M%S')}_{model_type}"
        
        # Initialize metrics storage
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # System info
        self.system_info = self._get_system_info()
        
        print(f"🌱 Plantesa Training Logger initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Model Type: {model_type}")
        print(f"   Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'python_version': platform.python_version(),
            'os_info': f"{platform.system()} {platform.release()}",
            'cpu_count': psutil.cpu_count(),
            'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        if TF_AVAILABLE:
            info['tensorflow_version'] = tf.__version__
            
            # Check for GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info['gpu_name'] = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'Unknown GPU')
                info['cuda_version'] = tf.config.experimental.get_device_details(gpus[0]).get('compute_capability', 'Unknown')
            else:
                info['gpu_name'] = "CPU Only"
                info['cuda_version'] = "N/A"
        else:
            info['tensorflow_version'] = "Not Available"
            info['gpu_name'] = "Unknown"
            info['cuda_version'] = "N/A"
        
        return info
    
    def log_epoch(self, epoch: int, logs: Dict[str, float], learning_rate: float = None):
        """
        Log metrics for each epoch
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics from Keras callback
            learning_rate: Current learning rate
        """
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(logs.get('loss', 0))
        self.metrics['train_accuracy'].append(logs.get('accuracy', 0))
        self.metrics['val_loss'].append(logs.get('val_loss', 0))
        self.metrics['val_accuracy'].append(logs.get('val_accuracy', 0))
        self.metrics['learning_rate'].append(learning_rate or 0)
        
        # Print formatted epoch summary
        print(f"\n📊 Epoch {epoch:03d} | "
              f"Loss: {logs.get('loss', 0):.4f} | "
              f"Acc: {logs.get('accuracy', 0):.4f} | "
              f"Val Loss: {logs.get('val_loss', 0):.4f} | "
              f"Val Acc: {logs.get('val_accuracy', 0):.4f}", end="")
        
        if learning_rate:
            print(f" | LR: {learning_rate:.6f}", end="")
    
    def create_run_entry(self, 
                         model,
                         dataset_info: Dict[str, Any],
                         hyperparams: Dict[str, Any],
                         final_metrics: Dict[str, float],
                         classification_report: Optional[Dict[str, Any]] = None,
                         confusion_matrix: Optional[List[List[int]]] = None,
                         training_time_minutes: float = 0,
                         early_stopped: bool = False,
                         best_epoch: int = 0,
                         teacher_model: Optional[str] = None,
                         distillation_params: Optional[Dict[str, float]] = None) -> TrainingRun:
        """
        Create a TrainingRun entry from collected data
        
        Args:
            model: Keras model instance
            dataset_info: Dataset information (name, size, num_classes, class_names)
            hyperparams: Hyperparameters (batch_size, image_size, epochs, learning_rate, etc.)
            final_metrics: Final metrics (loss, accuracy, etc.)
            classification_report: Classification report from sklearn
            confusion_matrix: Confusion matrix as list of lists
            training_time_minutes: Total training time in minutes
            early_stopped: Whether training was early stopped
            best_epoch: Best epoch number
            teacher_model: Teacher model name (for student models)
            distillation_params: Knowledge distillation parameters
        
        Returns:
            TrainingRun object
        """
        
        # Extract model architecture info
        model_architecture = "CNN"  # Default
        model_size = None
        pretrained = False
        
        # Try to detect model type from layers
        if hasattr(model, 'layers'):
            for layer in model.layers:
                layer_class = layer.__class__.__name__
                if 'VGG' in layer_class:
                    model_architecture = "VGG"
                    if '16' in layer_class:
                        model_size = "16"
                    elif '19' in layer_class:
                        model_size = "19"
                    pretrained = True
                    break
                elif 'ConvNeXt' in layer_class:
                    model_architecture = "ConvNeXt"
                    if 'Base' in layer_class:
                        model_size = "Base"
                    pretrained = True
                    break
        
        # Create run entry
        run = TrainingRun(
            # Identification
            run_id=self.run_id,
            date=self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            experiment_name=self.experiment_name,
            
            # Model Info
            model_type=self.model_type,
            model_architecture=model_architecture,
            model_size=model_size,
            pretrained=pretrained,
            
            # Dataset Info
            dataset_name=dataset_info.get('name', 'tomatoDataset(Augmented)'),
            dataset_size=dataset_info.get('size', 0),
            num_classes=dataset_info.get('num_classes', 10),
            class_names=dataset_info.get('class_names'),
            
            # Hyperparameters
            batch_size=hyperparams.get('batch_size', 32),
            image_size=hyperparams.get('image_size', 256),
            epochs_planned=hyperparams.get('epochs', 50),
            epochs_completed=len(self.metrics['epoch']),
            learning_rate=hyperparams.get('learning_rate', 0.001),
            optimizer=hyperparams.get('optimizer', 'Adam'),
            weight_decay=hyperparams.get('weight_decay', 0.0),
            
            # Performance Metrics
            train_accuracy=final_metrics.get('train_accuracy', 0),
            val_accuracy=final_metrics.get('val_accuracy', 0),
            test_accuracy=final_metrics.get('test_accuracy', 0),
            train_loss=final_metrics.get('train_loss', 0),
            val_loss=final_metrics.get('val_loss', 0),
            test_loss=final_metrics.get('test_loss', 0),
            
            # Classification Metrics
            precision=classification_report.get('precision', 0) if classification_report else 0,
            recall=classification_report.get('recall', 0) if classification_report else 0,
            f1_score=classification_report.get('f1_score', 0) if classification_report else 0,
            macro_avg_precision=classification_report.get('macro_avg', {}).get('precision', 0) if classification_report else 0,
            macro_avg_recall=classification_report.get('macro_avg', {}).get('recall', 0) if classification_report else 0,
            macro_avg_f1=classification_report.get('macro_avg', {}).get('f1-score', 0) if classification_report else 0,
            weighted_avg_precision=classification_report.get('weighted_avg', {}).get('precision', 0) if classification_report else 0,
            weighted_avg_recall=classification_report.get('weighted_avg', {}).get('recall', 0) if classification_report else 0,
            weighted_avg_f1=classification_report.get('weighted_avg', {}).get('f1-score', 0) if classification_report else 0,
            
            # Confusion Matrix
            confusion_matrix=confusion_matrix,
            
            # Training Metadata
            training_time_minutes=training_time_minutes,
            early_stopped=early_stopped,
            best_epoch=best_epoch,
            
            # System Info
            gpu_name=self.system_info['gpu_name'],
            gpu_memory_peak_gb=0.0,  # Would need to track during training
            cpu_count=self.system_info['cpu_count'],
            ram_total_gb=self.system_info['ram_total_gb'],
            python_version=self.system_info['python_version'],
            tensorflow_version=self.system_info['tensorflow_version'],
            cuda_version=self.system_info['cuda_version'],
            os_info=self.system_info['os_info'],
            
            # Paths
            model_path=hyperparams.get('model_path', ''),
            results_path=hyperparams.get('results_path', ''),
            config_path=hyperparams.get('config_path', ''),
            
            # Knowledge Distillation
            teacher_model=teacher_model,
            distillation_temperature=distillation_params.get('temperature', 1.0) if distillation_params else 1.0,
            distillation_alpha=distillation_params.get('alpha', 0.5) if distillation_params else 0.5
        )
        
        return run
    
    def save_metrics(self, filepath: str):
        """Save training metrics to JSON file"""
        metrics_path = Path(filepath)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"✓ Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Error saving metrics: {e}")

def main():
    """Main function for testing"""
    parser = argparse.ArgumentParser(description='Train model with logging')
    parser.add_argument('--experiment-name', type=str, required=True, help='Experiment name')
    parser.add_argument('--model-type', type=str, default='teacher', choices=['teacher', 'student'], help='Model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = TrainingLogger(args.experiment_name, args.model_type)
    
    print(f"\n🌱 Starting training: {args.experiment_name}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    
    # Example usage (would be integrated with actual training)
    print("\n💡 To integrate with your training loop:")
    print("   logger.log_epoch(epoch, logs, learning_rate)")
    print("   run = logger.create_run_entry(...)")
    print("   registry.add_run(run)")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Training Registry for Plantesa Leaf Disease Detection
Adapted for Keras/TensorFlow classification models
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import platform
import psutil

@dataclass
class TrainingRun:
    """Data class for storing training experiment information"""
    
    # Identification
    run_id: str
    date: str
    experiment_name: str
    
    # Model Information
    model_type: str  # 'teacher' or 'student'
    model_architecture: str  # CNN, VGG16, VGG19, ConvNeXtBase
    model_size: Optional[str] = None  # For VGG: '16', '19', etc.
    pretrained: bool = False
    
    # Dataset Information
    dataset_name: str = "tomatoDataset(Augmented)"
    dataset_size: int = 0
    num_classes: int = 10
    class_names: Optional[List[str]] = None
    
    # Training Hyperparameters
    batch_size: int = 32
    image_size: int = 256
    epochs_planned: int = 50
    epochs_completed: int = 0
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    weight_decay: float = 0.0
    
    # Performance Metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    test_loss: float = 0.0
    
    # Classification Metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    macro_avg_precision: float = 0.0
    macro_avg_recall: float = 0.0
    macro_avg_f1: float = 0.0
    weighted_avg_precision: float = 0.0
    weighted_avg_recall: float = 0.0
    weighted_avg_f1: float = 0.0
    
    # Confusion Matrix (stored as flattened list)
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Training Metadata
    training_time_minutes: float = 0.0
    early_stopped: bool = False
    best_epoch: int = 0
    
    # System Information
    gpu_name: str = "Unknown"
    gpu_memory_peak_gb: float = 0.0
    cpu_count: int = 0
    ram_total_gb: float = 0.0
    python_version: str = ""
    tensorflow_version: str = ""
    cuda_version: str = ""
    os_info: str = ""
    
    # File Paths
    model_path: str = ""
    results_path: str = ""
    config_path: str = ""
    
    # Status
    status: str = "completed"  # completed, failed, running
    
    # Knowledge Distillation (if student model)
    teacher_model: Optional[str] = None
    distillation_temperature: float = 1.0
    distillation_alpha: float = 0.5

class TrainingRegistry:
    """Registry for managing training experiment history"""
    
    def __init__(self, registry_path: str = "model/training_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs = self._load_registry()
    
    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load registry from JSON file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                return []
        return []
    
    def _save_registry(self):
        """Save registry to JSON file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.runs, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def add_run(self, run: TrainingRun):
        """Add a training run to the registry"""
        run_dict = asdict(run)
        
        # Convert class_names list to string if it exists
        if run_dict.get('class_names'):
            run_dict['class_names'] = list(run_dict['class_names'])
        
        # Convert confusion matrix to list of lists if it exists
        if run_dict.get('confusion_matrix'):
            run_dict['confusion_matrix'] = run.confusion_matrix
        
        self.runs.append(run_dict)
        self._save_registry()
        print(f"✓ Training run logged: {run.run_id}")
    
    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get all training runs"""
        return self.runs
    
    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific run by ID"""
        for run in self.runs:
            if run['run_id'] == run_id:
                return run
        return None
    
    def export_to_csv(self, csv_path: str = "model/training_history.csv"):
        """Export registry to CSV format"""
        if not self.runs:
            print("No runs to export")
            return
        
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV columns
        columns = [
            'run_id', 'date', 'experiment_name', 'model_type', 'model_architecture',
            'model_size', 'pretrained', 'dataset_name', 'dataset_size', 'num_classes',
            'batch_size', 'image_size', 'epochs_planned', 'epochs_completed',
            'learning_rate', 'optimizer', 'weight_decay', 'train_accuracy',
            'val_accuracy', 'test_accuracy', 'train_loss', 'val_loss', 'test_loss',
            'precision', 'recall', 'f1_score', 'macro_avg_precision',
            'macro_avg_recall', 'macro_avg_f1', 'weighted_avg_precision',
            'weighted_avg_recall', 'weighted_avg_f1', 'training_time_minutes',
            'early_stopped', 'best_epoch', 'gpu_name', 'gpu_memory_peak_gb',
            'status', 'teacher_model', 'distillation_temperature',
            'distillation_alpha'
        ]
        
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                
                for run in self.runs:
                    # Filter to only include columns that exist
                    row = {col: run.get(col, '') for col in columns}
                    writer.writerow(row)
            
            print(f"✓ Registry exported to CSV: {csv_path}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
    
    def generate_summary_table(self, md_path: str = "model/training_summary.md"):
        """Generate markdown summary table"""
        if not self.runs:
            print("No runs to summarize")
            return
        
        md_path = Path(md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(md_path, 'w') as f:
                f.write("# Plantesa Training Summary\n\n")
                
                # Group by date
                from collections import defaultdict
                runs_by_date = defaultdict(list)
                for run in self.runs:
                    date = run['date'].split()[0]
                    runs_by_date[date].append(run)
                
                for date, runs in sorted(runs_by_date.items(), reverse=True):
                    f.write(f"## {date}\n\n")
                    f.write("| Run ID | Model | Type | Batch | Size | LR/Epochs | Optimizer | Epochs | Train Acc | Val Acc | Test Acc | Training Time | GPU |\n")
                    f.write("|--------|-------|------|-------|------|-----------|-----------|--------|-----------|---------|----------|---------------|-----|\n")
                    
                    for run in runs:
                        run_id = run['run_id'][:8]
                        model = f"{run['model_architecture']}"
                        if run.get('model_size'):
                            model += f"-{run['model_size']}"
                        model_type = run.get('model_type', 'teacher')
                        batch = run['batch_size']
                        size = f"{run['image_size']}x{run['image_size']}"
                        lr_epochs = f"{run['learning_rate']}/{run['epochs_planned']}"
                        optimizer = run['optimizer']
                        epochs = f"{run['epochs_completed']}/{run['epochs_planned']}"
                        train_acc = f"{run['train_accuracy']:.3f}"
                        val_acc = f"{run['val_accuracy']:.3f}"
                        test_acc = f"{run['test_accuracy']:.3f}"
                        time_str = f"{run['training_time_minutes']:.1f}m"
                        gpu = run['gpu_name'].split()[0] if run['gpu_name'] != 'Unknown' else 'CPU'
                        
                        f.write(f"| {run_id} | {model} | {model_type} | {batch} | {size} | {lr_epochs} | {optimizer} | {epochs} | {train_acc} | {val_acc} | {test_acc} | {time_str} | {gpu} |\n")
                    
                    f.write("\n")
            
            print(f"✓ Markdown summary generated: {md_path}")
        except Exception as e:
            print(f"Error generating markdown: {e}")

def get_registry(registry_path: str = "model/training_registry.json") -> TrainingRegistry:
    """Get or create a training registry instance"""
    return TrainingRegistry(registry_path)

# Global registry instance
_registry_instance = None

def get_global_registry() -> TrainingRegistry:
    """Get global registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = TrainingRegistry()
    return _registry_instance
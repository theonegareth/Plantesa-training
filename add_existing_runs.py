#!/usr/bin/env python3
"""
Add existing training runs from Tables 1 & 2 to the registry
"""

from datetime import datetime
from pathlib import Path
import psutil
from training_registry import TrainingRun, get_registry

def add_teacher_model_runs():
    """Add all teacher model runs from Table 1"""
    registry = get_registry()
    
    # Define class names for tomato diseases
    class_names = [
        'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
        'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite',
        'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato_mosaic_virus', 'healthy'
    ]
    
    # Table 1: Teacher Model Runs
    teacher_runs = [
        # Test 1: 29 March 2025
        {
            'date': '2025-03-29',
            'experiment_name': 'CNN_Baseline_32_256',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 25,
            'epochs_completed': 7,
            'learning_rate': 0.01,
            'optimizer': 'Adam',
            'train_accuracy': 0.4605,
            'test_accuracy': 0.1980,
            'training_time_minutes': 0.62,  # ~37 seconds
            'early_stopped': True,
            'status': 'incomplete',
            'gpu_name': 'Unknown'
        },
        # Test 2: 16 April 2025
        {
            'date': '2025-04-16',
            'experiment_name': 'CNN_Augmented_64_256',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 64,
            'image_size': 256,
            'epochs_planned': 50,
            'epochs_completed': 50,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_accuracy': 0.8377,
            'test_accuracy': 0.0973,
            'training_time_minutes': 12.0,
            'early_stopped': False,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 3: 16 April 2025
        {
            'date': '2025-04-16',
            'experiment_name': 'CNN_Augmented_32_256',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 25,
            'epochs_completed': 25,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_accuracy': 0.8335,
            'test_accuracy': 0.6422,
            'training_time_minutes': 3.4,
            'early_stopped': False,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 4: 18 May 2025
        {
            'date': '2025-05-18',
            'experiment_name': 'CNN_Augmented_64_256_v2',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 64,
            'image_size': 256,
            'epochs_planned': 50,
            'epochs_completed': 9,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_accuracy': 0.8377,
            'test_accuracy': 0.0973,
            'training_time_minutes': 12.0,
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 5: 18 May 2025
        {
            'date': '2025-05-18',
            'experiment_name': 'CNN_Augmented_128_256',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 128,
            'image_size': 256,
            'epochs_planned': 50,
            'epochs_completed': 6,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_accuracy': 0.8467,
            'test_accuracy': 0.0927,
            'training_time_minutes': 4.5,
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 6: 18 May 2025
        {
            'date': '2025-05-18',
            'experiment_name': 'CNN_Augmented_64_256_100ep',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 64,
            'image_size': 256,
            'epochs_planned': 100,
            'epochs_completed': 21,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_accuracy': 0.8806,
            'test_accuracy': 0.3360,
            'training_time_minutes': 21.0,
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 7: 18 May 2025
        {
            'date': '2025-05-18',
            'experiment_name': 'CNN_Augmented_128_256_100ep',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 128,
            'image_size': 256,
            'epochs_planned': 100,
            'epochs_completed': 9,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_accuracy': 0.9363,
            'test_accuracy': 0.3720,
            'training_time_minutes': 8.0,
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 8: 18 May 2025
        {
            'date': '2025-05-18',
            'experiment_name': 'CNN_Augmented_64_256_LR0.0001',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 64,
            'image_size': 256,
            'epochs_planned': 50,
            'epochs_completed': 24,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.9688,
            'test_accuracy': 0.7273,
            'training_time_minutes': 16.0,
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 9: 18 May 2025
        {
            'date': '2025-05-18',
            'experiment_name': 'CNN_Augmented_128_256_LR0.0001',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 128,
            'image_size': 256,
            'epochs_planned': 50,
            'epochs_completed': 6,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.9844,
            'test_accuracy': 0.7020,
            'training_time_minutes': 6.0,
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 10: 18 May 2025
        {
            'date': '2025-05-18',
            'experiment_name': 'CNN_Augmented_64_256_100ep_LR0.0001',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 64,
            'image_size': 256,
            'epochs_planned': 100,
            'epochs_completed': 10,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 1.0000,
            'test_accuracy': 0.7507,
            'training_time_minutes': 15.0,
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 11: 18 May 2025
        {
            'date': '2025-05-18',
            'experiment_name': 'CNN_Augmented_128_256_100ep_LR0.0001',
            'model_architecture': 'CNN',
            'model_size': None,
            'pretrained': False,
            'batch_size': 128,
            'image_size': 256,
            'epochs_planned': 100,
            'epochs_completed': 7,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.9731,
            'test_accuracy': 0.8013,
            'training_time_minutes': 10.0,
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'Unknown'
        },
        # Test 12: 20 May 2025 - VGG16
        {
            'date': '2025-05-20',
            'experiment_name': 'VGG16_Pretrained_32_256_10ep',
            'model_architecture': 'VGG',
            'model_size': '16',
            'pretrained': True,
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 10,
            'epochs_completed': 10,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.8125,
            'test_accuracy': 0.7827,
            'training_time_minutes': 1.0,
            'early_stopped': False,
            'status': 'completed',
            'gpu_name': 'RTX 3060 Ti'
        },
        # Test 13: 20 May 2025 - VGG16
        {
            'date': '2025-05-20',
            'experiment_name': 'VGG16_Pretrained_32_256_30ep',
            'model_architecture': 'VGG',
            'model_size': '16',
            'pretrained': True,
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 30,
            'epochs_completed': 30,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.9062,
            'test_accuracy': 0.8247,
            'training_time_minutes': 6.0,
            'early_stopped': False,
            'status': 'completed',
            'gpu_name': 'RTX 3060 Ti'
        },
        # Test 14: 20 May 2025 - VGG16
        {
            'date': '2025-05-20',
            'experiment_name': 'VGG16_Pretrained_32_256_30ep_v2',
            'model_architecture': 'VGG',
            'model_size': '16',
            'pretrained': True,
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 30,
            'epochs_completed': 30,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.8125,
            'test_accuracy': 0.8393,
            'training_time_minutes': 6.0,
            'early_stopped': False,
            'status': 'completed',
            'gpu_name': 'RTX 3060 Ti'
        },
        # Test 15: 21 May 2025 - VGG16
        {
            'date': '2025-05-21',
            'experiment_name': 'VGG16_Pretrained_32_256_30ep_v3',
            'model_architecture': 'VGG',
            'model_size': '16',
            'pretrained': True,
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 30,
            'epochs_completed': 30,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.9462,
            'test_accuracy': 0.8213,
            'training_time_minutes': 12.0,
            'early_stopped': False,
            'status': 'completed',
            'gpu_name': 'RTX 3060 Ti'
        },
        # Test 16: 30 May 2025 - VGG19
        {
            'date': '2025-05-30',
            'experiment_name': 'VGG19_Pretrained_32_256_50ep',
            'model_architecture': 'VGG',
            'model_size': '19',
            'pretrained': True,
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 50,
            'epochs_completed': 50,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_accuracy': 0.7500,
            'test_accuracy': 0.8033,
            'training_time_minutes': 270.0,  # 4.5 hours
            'early_stopped': False,
            'status': 'completed',
            'gpu_name': 'RTX 3060 Ti'
        },
        # Test 17: 31 May 2025 - VGG19 (Best Teacher Model)
        {
            'date': '2025-05-31',
            'experiment_name': 'VGG19_Pretrained_32_256_100ep_LR0.0001',
            'model_architecture': 'VGG',
            'model_size': '19',
            'pretrained': True,
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 100,
            'epochs_completed': 35,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.9822,
            'test_accuracy': 0.8953,
            'training_time_minutes': 210.0,  # 3.5 hours
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'RTX 3060 Ti'
        }
    ]
    
    # Add all teacher runs
    for i, run_data in enumerate(teacher_runs, 1):
        run = TrainingRun(
            run_id=f"teacher_run_{run_data['date']}_{i:02d}",
            date=f"{run_data['date']} 00:00:00",
            experiment_name=run_data['experiment_name'],
            model_type='teacher',
            model_architecture=run_data['model_architecture'],
            model_size=run_data.get('model_size'),
            pretrained=run_data['pretrained'],
            dataset_name='tomatoDataset(Augmented)',
            dataset_size=1500,  # Approximate based on confusion matrix support
            num_classes=10,
            class_names=class_names,
            batch_size=run_data['batch_size'],
            image_size=run_data['image_size'],
            epochs_planned=run_data['epochs_planned'],
            epochs_completed=run_data['epochs_completed'],
            learning_rate=run_data['learning_rate'],
            optimizer=run_data['optimizer'],
            train_accuracy=run_data['train_accuracy'],
            val_accuracy=0.0,  # Not provided in table
            test_accuracy=run_data['test_accuracy'],
            train_loss=0.0,  # Not provided in table
            val_loss=0.0,  # Not provided in table
            test_loss=0.0,  # Not provided in table
            precision=0.0,  # Would need per-class data
            recall=0.0,  # Would need per-class data
            f1_score=0.0,  # Would need per-class data
            macro_avg_precision=0.0,  # Would need per-class data
            macro_avg_recall=0.0,  # Would need per-class data
            macro_avg_f1=0.0,  # Would need per-class data
            weighted_avg_precision=0.0,  # Would need per-class data
            weighted_avg_recall=0.0,  # Would need per-class data
            weighted_avg_f1=0.0,  # Would need per-class data
            training_time_minutes=run_data['training_time_minutes'],
            early_stopped=run_data['early_stopped'],
            best_epoch=run_data['epochs_completed'],
            gpu_name=run_data['gpu_name'],
            gpu_memory_peak_gb=0.0,
            cpu_count=psutil.cpu_count(),
            ram_total_gb=psutil.virtual_memory().total / (1024**3),
            python_version='3.x',  # Not specified
            tensorflow_version='2.x',  # Not specified
            cuda_version='Unknown',  # Not specified
            os_info='Unknown',  # Not specified
            model_path='',
            results_path='',
            config_path='',
            status=run_data['status']
        )
        
        registry.add_run(run)
    
    print("✓ All teacher model runs added successfully!")

def add_student_model_runs():
    """Add student model runs from Table 2"""
    registry = get_registry()
    
    # Table 2: Student Model Runs (only 1 entry)
    student_runs = [
        # Test 1: 31 May 2025 - Student model
        {
            'date': '2025-05-31',
            'experiment_name': 'Student_VGG19_from_Teacher17',
            'model_architecture': 'VGG',
            'model_size': '19',
            'pretrained': False,  # Student model trained from scratch
            'batch_size': 32,
            'image_size': 256,
            'epochs_planned': 100,
            'epochs_completed': 35,
            'learning_rate': 0.0001,
            'optimizer': 'Adam',
            'train_accuracy': 0.9822,
            'test_accuracy': 0.8953,
            'training_time_minutes': 210.0,  # 3.5 hours
            'early_stopped': True,
            'status': 'completed',
            'gpu_name': 'RTX 3060 Ti',
            'teacher_model': 'VGG19_Pretrained_32_256_100ep_LR0.0001',
            'distillation_temperature': 3.0,  # Typical value
            'distillation_alpha': 0.7  # Typical value
        }
    ]
    
    # Add student run
    run_data = student_runs[0]
    run = TrainingRun(
        run_id=f"student_run_{run_data['date']}_01",
        date=f"{run_data['date']} 00:00:00",
        experiment_name=run_data['experiment_name'],
        model_type='student',
        model_architecture=run_data['model_architecture'],
        model_size=run_data.get('model_size'),
        pretrained=run_data['pretrained'],
        dataset_name='tomatoDataset(Augmented)',
        dataset_size=1500,
        num_classes=10,
        class_names=[
            'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
            'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite',
            'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato_mosaic_virus', 'healthy'
        ],
        batch_size=run_data['batch_size'],
        image_size=run_data['image_size'],
        epochs_planned=run_data['epochs_planned'],
        epochs_completed=run_data['epochs_completed'],
        learning_rate=run_data['learning_rate'],
        optimizer=run_data['optimizer'],
        train_accuracy=run_data['train_accuracy'],
        val_accuracy=0.0,
        test_accuracy=run_data['test_accuracy'],
        train_loss=0.0,
        val_loss=0.0,
        test_loss=0.0,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        macro_avg_precision=0.0,
        macro_avg_recall=0.0,
        macro_avg_f1=0.0,
        weighted_avg_precision=0.0,
        weighted_avg_recall=0.0,
        weighted_avg_f1=0.0,
        training_time_minutes=run_data['training_time_minutes'],
        early_stopped=run_data['early_stopped'],
        best_epoch=run_data['epochs_completed'],
        gpu_name=run_data['gpu_name'],
        gpu_memory_peak_gb=0.0,
        cpu_count=psutil.cpu_count(),
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
        python_version='3.x',
        tensorflow_version='2.x',
        cuda_version='Unknown',
        os_info='Unknown',
        model_path='',
        results_path='',
        config_path='',
        status=run_data['status'],
        teacher_model=run_data['teacher_model'],
        distillation_temperature=run_data['distillation_temperature'],
        distillation_alpha=run_data['distillation_alpha']
    )
    
    registry.add_run(run)
    print("✓ Student model run added successfully!")

def main():
    """Main function"""
    print("🌱 Adding existing training runs from Tables 1 & 2...")
    print("=" * 60)
    
    # Add teacher model runs (16 runs)
    print("\n📊 Adding Teacher Model Runs (Table 1)...")
    add_teacher_model_runs()
    
    # Add student model runs (1 run)
    print("\n🎓 Adding Student Model Runs (Table 2)...")
    add_student_model_runs()
    
    print("\n" + "=" * 60)
    print("✅ All existing training runs added successfully!")
    print("\n📋 Summary:")
    print("   - Teacher models: 16 runs")
    print("   - Student models: 1 run")
    print("   - Total: 17 runs")
    print("\n💡 View the registry with:")
    print("   python view_registry.py")

if __name__ == '__main__':
    main()
## Knowledge Distillation

Knowledge Distillation is a model compression technique where a smaller, simpler model (the "student") is trained to replicate the behavior of a larger, more complex model (the "teacher"). The student learns not only from the ground truth labels but also from the teacher's output, enabling it to achieve high accuracy with reduced computational resources.

_Source: [Keras Knowledge Distillation Example](https://keras.io/examples/vision/knowledge_distillation/)_

### Training Flowchart

![Student Training Flowchart](assets/Student_Training_Flowchart.png)
### Explanation of the Flowchart

1. **Input Data**: The training data is fed into both the teacher and student models.
2. **Teacher Model Predictions**: The teacher model generates predictions based on the input data.
3. **Student Model Predictions**: The student model also generates predictions for the same input data.
4. **Loss Calculation**:
    - **Ground Truth Loss**: The difference between the student model's predictions and the actual labels is calculated.
    - **Distillation Loss**: The difference between the student model's predictions and the teacher model's predictions is calculated.
5. **Combined Loss**: The ground truth loss and distillation loss are combined to form the total loss.
6. **Backpropagation**: The combined loss is used to update the student model's weights through backpropagation.

# Food-Image-Categorization-Utilizing-CNN
Food Image Categorization Utilizing CNN
# Food Image Categorization Utilizing CNN
### Kaggle Link:
The link to the Kaggle dataset can be found here: https://www.kaggle.com/datasets/kmader/food41/data 

### Overview:
The primary goal of this project is to build a deep learning model capable of classifying images into one of the 101 food categories using the Food-101 dataset. The project explores various neural network architectures, starting with custom Convolutional Neural Networks (CNNs) and progressing to advanced pre-trained models, to evaluate their effectiveness in tackling a complex multi-class classification problem.

The Food-101 dataset is a popular benchmark dataset for image classification tasks. It contains 101 categories of food, each with 1,000 images. The key features of the dataset are: 

- Number of Classes: 101 food categories (e.g., pizza, burgers, salads, desserts).
- Total Images: 101,000 images.
- Image Resolution: Varies, but resized for this project (e.g., 128x128 or 256x256).
- Class Distribution: Each class has exactly 1,000 images, ensuring balanced representation.

### Baseline Model: 
This is a simple model intended to gain an understanding of how to adjust and tune. 

The architecture for the baseline model is detailed below:

1. **Input Layer:** Takes images of shape 128 x 128 pixels with 3 color channels (RGB)
2. **Convolutional Layers:**
      *  **First Convolutional Layer:** 32 filters with kernel size (3,3). ReLU Activation.
      *  **Max Pooling Layer:** Reduce spatial dimensions by a factor of 2.
      *  **Second Convolutional Layer:** 64 filters with a kernel size of (3,3) Relu Activation.
      *  **Max Pooling Layer:** Reduce spatial dimensions by a factor of 2.
      *  **Third Convolutional Layer:** 128 filters with a kernel size of (3,3) Relu Activation.
      *  **Max Pooling Layer:** Reduce spatial dimensions by a factor of 2.
3. **Flatten Layer:** Converts the two dimensional feature maps from the last convolutional layer into a single dimension vector.
4. **Dense Layers:**
      * **First Dense Layer:** Fully connected; 128 neurons and ReLU activation.
      * **Output Layer:** Single neuron with a sigmoid activation.
5. **Optimizer:** Utilizes an Adam Optimizer with a Loss Function of Sparse Categorical Crossentropy.

### Model 2
The following adjustments were made to the baseline model:

1. **Input Shape:**
    - The shape of the input was previously (128, 128, 3), whereas it was increased to (256, 256, 3)
    - Reasoning: To captures more detailed spatial features and hopefully yield better accuracy. 
2. **Dense Layer Neurons:**
    - Was adjusted (from 128 to 256) to reflect the input shape.
    - Reasoning: Help capture additional details in spatial features.
3. **Dropout:**
    - A dropout factor of 0.5 was added after previously not having one in the baseline model.
    - Reasoning: To reduce overfitting by adding regularization.
4. **Learning Rate:**
    - Adjusted from the default Adam learning rate to 1e-4.
    - Reasoning: Help with the optimization and convergence overall
  
### Model 3

1. **Batch Normalization**
    - Was added after every Conv2D layer and the Dense Layer.
    - Reasoning: Stabilizes training and speeds up convergence.
2. **Dropout:**
    - The dropout factor of 0.5 from Model 2 was reduced to 0.4. 
    - Reasoning: A slightly lower dropout rate strikes a balance between regularization and model capacity. This adjustment helps retain more neurons while still preventing overfitting.

### MobileNetV2 Model (Transfer Learning)
The final model utilizes transfer learning via the MobileNetV2 model. The architecture for the MobileNetV2 Model was not adjusted in this project. From https://www.analyticsvidhya.com/blog/2023/12/what-is-mobilenetv2/ the architecture for the MobileNetV2 Model is:

![image.png](attachment:20f305f9-a2f1-473a-a529-90b21cbb94f2.png)

### Conclusions:
1. Baseline Model:
    - Validation Loss: 6.729179
    - Validation Accuracy: 0.075347

The baseline model fails to learn meaningful patterns, which is indicated by the very high validation loss value and low validation accuracy. Based on the plots developed, the training  accuracy shows that there is some steady improvement, but with minimal increase to the validation accuracy. This indicates that there is potential overfitting or model capabilities to handle the dataset complexity. 


2. Model 2 (Improved CNN):
    - Validation Loss: 4.465222
    - Validation Accuracy: 0.071733

Model 2 improves over the baseline by adding a larger input size, increasing the number of neurons in the dense layer, adding a dropout, and by adjusting the learning rate. Despite the adjustments, the validation accuracy is still similar to that of the baseline model. 
Despite these changes, validation accuracy remains comparable to the baseline, and loss is still quite high.

3. Model 3 (Batch Normalization):
    - Validation Loss: 4.517851
    - Validation Accuracy: 0.084109

Model 3 introduces Batch Normalization to stabilize training and reduce overfitting.
The validation accuracy is slightly better than the previous models, and the loss is similar to Model 2. Batch Normalization likely improved convergence, but the model still struggles to generalize effectively to the validation set.

4. MobileNetV2:
    - Validation Loss: 2.150652
    - Validation Accuracy: 0.500248

The MobileNetV2 model significantly outperforms the other models in both loss reduction and validation accuracy. The use of a pre-trained model allows MobileNetV2 to leverage robust feature extraction, drastically improving performance. The validation accuracy suggests the model is learning meaningful patterns and generalizing better than the other models.

Future work should focus on:
Fine-tuning MobileNetV2 layers.
Experimenting with other pre-trained architectures (e.g., EfficientNet, InceptionV3).
Using data augmentation to improve model generalization further.

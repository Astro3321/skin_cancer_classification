# Skin Cancer Classification

This Project focuses on developing a Deep Learning Model for the classification of Skin Cancer. The Approach used to solve this problem is a 6-layer CNN Architecture with 2 Dense Layers, carefully designed to improve the learning capabilities of the Model.

## Architecture
The CNN architecture consists of four Convolutional Layers with progressively decreasing filter sizes. The first two Convolutional Layers have 64 filters each, followed by two more Convolutional Layers with 32 filters each. All convolutional layers have a kernel size of 3x3. The Convolutional Layers are interconnected with Max-Pooling Layers and Dropout Layers to reduce dimensionality and add regularization to the model.

The Output of the Convolutional Layers is flattened using a Flatten Layer, which converts the output matrix into a single-dimensional array. This flattened output is then fed into two Dense Layers. The first Dense Layer has 256 units and is followed by the output Dense Layer with 7 units, representing each class. Rectified Linear Unit (ReLU) is the activation function for all layers except the output layer, which utilizes the Softmax Activation function.

## Training
The Stratified KFold Algorithm is employed to train the Model with 5 folds. Each fold has an Epoch Cycle of 8 and a Batch Size of 32. This approach allows for cross-validation and enables the generation of multiple models with varying accuracies. The model is saved at each fold to obtain multiple models for ensemble learning.

The Loss Function used during training is Sparse Categorical Crossentropy, which is suitable for multi-class classification tasks. The Optimizer Algorithm employed is ADAM (Adaptive Moment Estimation), a popular optimization algorithm known for its efficiency in training deep neural networks.

## Ensemble Learning
To improve the accuracy and Generalization of the overall model, Ensemble learning methods are utilized for prediction. The models with the best accuracy are selected, and their predictions are combined to produce the final prediction. This ensemble approach harnesses the strengths of multiple models, leading to improved overall performance.

## Results
After testing and observing various Model Architectures for Skin Cancer Classification, the presented approach yielded the best results. By leveraging the power of Deep Learning and employing a carefully designed CNN architecture, the model demonstrates strong classification capabilities, contributing to the accurate identification of skin cancer types.

| Test Image                          | Prediction                           |
| ----------------------------------- | ----------------------------------- |
| ![Test Image](https://github.com/Astro3321/skin_cancer_classification/assets/69784938/79071d12-56e5-4861-a3ad-b87a5d55f5ec) | ![Prediction](https://github.com/Astro3321/skin_cancer_classification/assets/69784938/1567c249-403c-427e-9dac-8b545370f4b3) |

## Dependencies
- Python
- TensorFlow
- Keras
- CNN

## Conclusion
The Skin Cancer Classification project presents a robust Deep Learning Model that effectively identifies different types of Skin Cancer. The carefully designed CNN architecture, along with ensemble learning techniques, contributes to improved Accuracy and Generalization.

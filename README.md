# Heat-Transfer-in-Advanced-Manufacturing-using-PINN
This is a PINN based approach in solving high temperature heat transfer equations in manufacturing industries, with a focus on reducing the energy consumption and optimizing the sensor positioning.

## Overview

Physics-Informed Neural Networks (PINNs) are a novel class of neural networks that leverage physical laws described by partial differential equations (PDEs) to inform the learning process. This project implements a PINN to model the temperature distribution in a solidification process.

## Process Description

This project is based on the research paper [Machine learning for metal additive manufacturing: Predicting temperature and melt pool fluid dynamics using physics-informed neural networks](https://arxiv.org/abs/2008.13547) by Qiming Zhu, Zeliang Liu, Jinhui Yan.

The paper introduces the concept of Physics-Informed Neural Networks (PINNs), which embed physical laws into the learning process of neural networks. The methodology uses automatic differentiation to incorporate the PDEs into the loss function, guiding the training process with both data and physical laws.

### Adaptation to TensorFlow 2.11.0

The original implementation provided in the paper was designed for TensorFlow 1.x, which is not compatible with TensorFlow 2.x. This repository contains a refactored version of the code, making it compatible with TensorFlow 2.11.0. The key changes include:

1. **Session Management**: TensorFlow 2.x uses eager execution by default, removing the need for explicit session management. However, to maintain the structure of the original code, the `tf.compat.v1.Session` is used for session-based execution.
2. **Optimizers**: TensorFlow 2.x has a new API for optimizers. The code now uses `tf.keras.optimizers` and the `minimize` method to handle optimization.
3. **Gradient Computation**: The `tf.GradientTape` context is used for computing gradients in TensorFlow 2.x, replacing the `tf.gradients` function from TensorFlow 1.x.
4. **Eager Execution Compatibility**: Ensured all tensor operations are compatible with eager execution to facilitate debugging and model development.

These changes ensure that the code is up-to-date with the latest version of TensorFlow, benefiting from improved performance, ease of use, and ongoing support.

## Model Explanation

The PINN model is trained to solve the following PDE:

\[ \frac{\partial T}{\partial t} = \alpha \nabla^2 T \]

where \( T \) is the temperature, \( t \) is time, and \( \alpha \) is the thermal diffusivity.

### Loss Function

The loss function is a combination of the data loss and the physics-informed loss:
\[ \text{Loss} = \text{MSE}(\hat{T}, T) + \lambda \cdot \text{PDE\_Loss} \]

### Neural Network Architecture

The model consists of 8 hidden layers with 300 nodes each, using the ReLU activation function.

### Training

The model is trained for 50,000 epochs using the Adam optimizer with a learning rate of 0.001.

### Results

The PINN model achieved a mean squared error (MSE) loss of 0.19. Below are some visualizations of the predictions compared to the exact solutions.



## Visualisation

The model predicts temperature in Kelvin, time in seconds, and X in metres. Below are some visualizations of the results.

![Exact vs Predicted Temperature](visualisation/final_results/Temp_Pred_VS_Exact.png)
*Scatter plot of exact vs predicted temperatures.*

![Residuals](visualisation/final_results/Residuals_Tem_Pred.png)
*Residual plot (Exact - Predicted temperatures).*

![Temperature Distribution](visualisation/final_results/Scatter_Plot.png)
*Temperature distribution over time.*

## Usage

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Installation

Clone the repository:
```sh
git clone https://github.com/your_username/your_repo_name.git
cd your_repo_name
# Phase 1: Neural Network Architecture & Backpropagation

This repository contains documentation on the fundamental mechanics of Neural Networks, from single-layer perceptrons to deep networks utilizing backpropagation.

## 1. Neural Network Fundamentals

A **neuron** is a computational unit that activates when it reaches a specific threshold, sending data to the next unit. A **Neural Network (NN)** is made up of these neurons connected by weighted edges. 
* By structuring these connections, networks mathematically map inputs to outputs. 
* The structure of the network and the number of neurons determine the complexity of this mapping. 
* The model "learns" based on this structure and training data by computing and adjusting the weights that determine the output for given inputs.

## 2. Activation Functions

The activation function determines the class and format of the outputs. Three common types exist:
* **Unit Step Function:** `h = 0` for `y < 0`, and `h = 1` for `y >= 0`.
* **Sigmoid Function:** `h = e^y / (1 + e^y)`. Outputs a continuous value between 0 and 1.
* **Rectified Linear Unit (ReLU):** `h = max(0, y)`. Outputs 0 if negative, and the raw value if positive.

## 3. The Learning Process (Gradient Descent)

The process of learning is called **Gradient Descent**. The model follows a 5-step loop to calculate and update weights:

1. **Initialize:** Assume small initial values for the weights.
2. **Forward Pass:** Compute the predicted output value by passing the inputs through the linear hypothesis function and then the activation function.
3. **Compute Loss:** Calculate the error. This depends on the specific loss function used, but a basic heuristic is `Loss = (h_pred - h_actual)`.
4. **Compute Gradients:** For each weight, calculate the gradient using the Chain Rule. 
   * `dL/dW = dL/dh * dh/dy * dy/dw`
5. **Update Weights:** Adjust the weights by stepping in the opposite direction of the gradient.
   * `w_new = w_old - (alpha * gradient)` *(where alpha is the learning rate/step size).*

## 4. Linear Separability (Single-Layer Networks)

A single-layer network is sufficient when the target outputs are **linearly separable** (a straight line can cleanly divide the classes).

* **The OR Logic Gate:** Suppose `y = w1*x1 + w2*x2 + b`. If `b = -1`, `w1 = 1`, and `w2 = 1` using a unit step activation function:
  * At (0,0), `h = 0`
  * At (0,1), `h = 1`
  * At (1,0), `h = 1`
  * At (1,1), `h = 1`
  A straight line separates the 0 and 1 classes perfectly.
* **The AND Logic Gate:** The same architecture works, but the bias is shifted (e.g., `b = -2`). The normal gradient descent method works perfectly here because the data allows for a clean linear split.

## 5. Non-Linear Separability (The XOR Problem)

For complex logic like the **XOR gate**, the outputs are *not* linearly separable. 
* Using the previous single-layer architecture (`y = w1*x1 + w2*x2 + b`), a single straight line cannot separate the classes. At least one output will always be classified incorrectly.
* **The Solution (Hidden Layers):** To solve this, a hidden layer is added. The network can combine an OR logic unit and an AND logic unit in the hidden layer, weight their outputs, and send them to a final output neuron.
  * `h_final = w5*h1 + w6*h2 + b3`
  * `h1 = Activation(w1*x1 + w2*x2 + b1)`
  * `h2 = Activation(w3*x1 + w4*x2 + b2)`
By combining these linear boundaries, the model learns the non-linear XOR pattern perfectly.

## 6. Backpropagation

When hidden layers are introduced to solve non-linear problems, a step called **Backpropagation** is added to the learning process. The model calculates the gradients by chaining derivatives backward from the output layer to the first layer.

**For a 2-Layer Network (1 Hidden, 1 Output):**
* **Output Layer Weight (w5):** `dL/dw5 = dL/dh * dh/dy3 * dy3/dw5`
* **Hidden Layer Weight (w1):** `dL/dw1 = dL/dh * dh/dy3 * dy3/dh1 * dh1/dy1 * dy1/dw1`
*(Note how the partial derivatives from the output layer are carried backward and included in the calculation for the earlier layers).*

**Example: A 4-Layer Network (1 Input, 2 Hidden, 1 Output):**
To calculate the gradient for a weight in the very first layer (`w1`), the chain rule extends through every layer:
* `dL/dw1 = dL/dh * dh/dy5 * dy5/dh3 * dh3/dy3 * dy3/dh1 * dh1/dy1 * dy1/dw1`
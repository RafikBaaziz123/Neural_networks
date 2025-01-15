# Neural Network for Animal Classification

## Project Overview
This project involves building and training a multilayer neural network capable of classifying animals into one of seven types based on their characteristics. The neural network is trained using data from the `zoo.txt` file, which contains details about 121 animals, each described by 18 attributes. The primary goal is to adapt existing code for neural network training to solve this classification problem.

## Objectives
1. **Learn to adapt and extend neural network training code** for a new classification problem.
2. **Implement functions** to process data and generate training and test sets.
3. **Train a neural network** to correctly identify animal types.
4. **Evaluate the performance** of the trained neural network.

## Features
- Neural network with backpropagation for training.
- Conversion of animal data into structured training patterns.
- Support for 7 animal types as outputs.
- Evaluation and accuracy calculation for test datasets.

## Dataset Description
The dataset is split into:
- **Input Attributes**: 16 numerical attributes (e.g., feathers, milk production, etc.).
- **Animal Type**: The final classification output with 7 possible types (e.g., mammal, reptile).

Each line in `zoo.txt` describes an animal:
- The first value is the animal's name.
- The next 16 values are numerical inputs.
- The last value is the animal type.

## Key Functions

### 1. `build_sets`
This function processes the dataset and creates:
- **Training Set**: First 67 patterns.
- **Test Set**: Remaining patterns.

Steps:
- Read and parse each line from `zoo.txt`.
- Convert attributes into binary input patterns.
- Generate output patterns based on animal types.
- Randomize the order of patterns.

### 2. `translate`
Converts individual animal data into a structured pattern:
- Input: `[name, attributes, type]`
- Output: `[animal_name, input_pattern, animal_type, output_pattern]`

Example:
Input: `["aardvark", 1, 0, 0, 1, ..., "mammal"]`
Output:
```python
['aardvark', [1, 0, 0, 1, 0, ..., 1], 'mammal', [1, 0, 0, 0, 0, 0, 0]]
```

### 3. `train_zoo`

- Trains the neural network for 300 iterations using the training set.

- Calls the iterate function to adjust weights for each training pattern.

### 4. `test_zoo`

- Tests the trained network using the test set.

- Outputs predictions and compares them with actual types.

- Calculates and prints the success rate.

- Example output:
```java
The network thinks mongoose is a mammal, it should be a mammal
Success rate: 94.12%
```

### 5. `retranslate`

- Converts neural network outputs into corresponding animal types.

## Implementation Steps

1. Analyze and understand the provided code for Boolean functions (AND, OR, XOR).

2. Implement the `build_sets` and `translate` functions to prepare training and test datasets.

3. Train the neural network using the `train_zoo` function.

4. Test the neural network using the `test_zoo` function and evaluate performance.

## Evaluation Metrics

- Accuracy: Percentage of correct classifications.

- Output Comparison: Detailed predictions for each test case.

## Example Results

- Example input: Attributes of an animal (binary format).

- Predicted output: `['mammal', 'reptile', ...]`.

- Success rate: e.g., `94.12%`.

## Prerequisites

- **Python**: Ensure Python is installed on your system.

- Required libraries: NumPy (or other dependencies mentioned in the provided code).

## How to Run

- Place the zoo.txt and info.txt files in the project directory.

- Run the script to process data and train the network.

- Use the test function to evaluate network performance.

## References

Course Material: Artificial Intelligence

Provided by: Mr. Arlindo Silva

Files: zoo.txt, info.txt, and sample neural network training code.

## Authors

Aziz Zina and Rafik Baaziz.

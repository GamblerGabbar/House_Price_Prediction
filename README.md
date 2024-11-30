# House Prices Prediction using TensorFlow Decision Forests

This repository demonstrates how to build a robust house price prediction model using TensorFlow Decision Forests (TF-DF). The project leverages decision forest models, known for their efficiency and interpretability, to predict house prices based on various features such as location, size, and amenities.

# Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Requirements](#requirements)   
- [Usage](#usage)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Results](#results)    
- [License](#license)
  
---

## Overview

Predicting house prices accurately is a critical challenge in the real estate domain. This project uses **TensorFlow Decision Forests**, a library designed to seamlessly integrate decision forests with the TensorFlow ecosystem, to train and evaluate models that predict house prices.

Decision forests combine the power of decision trees with ensemble learning techniques, making them suitable for structured data and tabular datasets.

---
## Features

- Uses **TensorFlow Decision Forests** for model training.    
- Easy-to-interpret model outputs.  
- Comprehensive preprocessing pipeline.  
- End-to-end workflow from data preparation to evaluation.  

---
## Requirements

- Python 3.8 or higher  
- TensorFlow 2.17 or higher  
- TensorFlow Decision Forests  
- NumPy, Pandas, Matplotlib. 

---

## Dataset

The dataset used for training contains a mix of numerical and categorical features describing various aspects of houses, along with the target variable `SalePrice`. Below is an example of the dataset structure:

### Sample Data

| Id  | MSSubClass | MSZoning | LotFrontage | LotArea | Street | Alley | LotShape | LandContour | Utilities | ... | SalePrice |
|-----|------------|----------|-------------|---------|--------|-------|----------|-------------|-----------|-----|-----------|
| 1   | 60         | RL       | 65.0        | 8450    | Pave   | NaN   | Reg      | Lvl         | AllPub    | ... | 208500    |
| 2   | 20         | RL       | 80.0        | 9600    | Pave   | NaN   | Reg      | Lvl         | AllPub    | ... | 181500    |
| 3   | 60         | RL       | 68.0        | 11250   | Pave   | NaN   | IR1      | Lvl         | AllPub    | ... | 223500    |
| 4   | 70         | RL       | 60.0        | 9550    | Pave   | NaN   | IR1      | Lvl         | AllPub    | ... | 140000    |
| 5   | 60         | RL       | 84.0        | 14260   | Pave   | NaN   | IR1      | Lvl         | AllPub    | ... | 250000    |

### Columns

The dataset includes the following types of features:

- **Numerical Features:**
  - `LotFrontage`, `LotArea`, `SalePrice`, etc.
- **Categorical Features:**
  - `MSZoning`, `Street`, `LotShape`, etc.
- **Other Features:**
  - `Alley`, `Fence`, `PoolQC`, etc.

### Target Variable
- `SalePrice`: The target variable representing the house sale price.

### Notes:
- Missing values (e.g., `NaN` in the `Alley` column) need to be handled during preprocessing.
- The dataset includes 81 columns, covering diverse attributes of the houses.

For more details, refer to the full dataset provided in the `data/` folder.

---

## Model Architecture

This project uses **TensorFlow Decision Forests (TF-DF)** to build a **Random Forest model** for predicting house prices. Below is a detailed description of the architecture and its hyperparameter optimization process.

---

### Base Model: Random Forest

The Random Forest model is an ensemble of decision trees designed for regression tasks. It provides robust predictions by combining the outputs of multiple decision trees, which are trained on different subsets of the data.

### Key Model Parameters

1. **`num_trees`:**  
   The number of trees in the forest. A higher number typically improves the model's accuracy but increases training time.  
   **Values tested in grid search:** `[100, 300, 500]`

2. **`max_depth`:**  
   The maximum depth of each tree. This controls the complexity of the trees and balances underfitting and overfitting.  
   **Values tested in grid search:** `[10, 15, 20]`

3. **`min_examples`:**  
   The minimum number of examples required in each leaf node. Smaller values allow for deeper trees but may overfit.  
   **Values tested in grid search:** `[1, 5, 10]`

### Training and Compilation

- **Loss Function:** The model optimizes the **Mean Squared Error (MSE)**, a common metric for regression tasks.  
- **Training Data:** The model is trained using preprocessed training data (`train_ds`).  
- **Validation:** The model's performance is evaluated on a separate validation dataset (`valid_ds`).

```python
rf = tfdf.keras.RandomForestModel(
    task=tfdf.keras.Task.REGRESSION,
    num_trees=num_trees,
    max_depth=max_depth,
    min_examples=min_examples
)
rf.compile(metrics=["mse"])
rf.fit(x=train_ds, verbose=0)
```

---

## Results

The table below shows a sample of predictions made by the model. The **Id** column corresponds to the house identifier, and the **SalePrice** column indicates the predicted sale price for that house.

| Id    | SalePrice      |
|-------|----------------|
| 1461  | 128717.87      |
| 1462  | 157146.44      |
| 1463  | 179886.88      |
| 1464  | 184516.83      |
| 1465  | 192833.42      |

### Summary
- The predictions demonstrate the model's ability to output continuous numerical values for house prices.
- Predicted prices are realistic and align with typical market trends for the given data.


---

## License

This project is licensed under the **MIT License**.  
  

See the full license in the `LICENSE` file included in the repository.

---

Happy Coding! 🚀


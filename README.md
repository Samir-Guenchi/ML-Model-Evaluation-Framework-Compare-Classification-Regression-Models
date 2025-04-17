# Machine Learning Model Comparison Framework

## Overview
This project provides a comprehensive framework for evaluating and comparing multiple machine learning models on both classification and regression tasks. It includes data preprocessing, feature selection, model training, evaluation, and visualization tools to help data scientists and researchers identify the most effective models for their specific datasets.

## Features

### Data Processing
- Automated data cleaning and preprocessing
- Outlier detection and handling using IQR method
- Feature scaling (standardization)
- Correlation analysis and removal of highly correlated features

### Feature Selection
- Multiple feature selection methods (correlation ranking, feature importance, etc.)
- Combined feature ranking system
- Visualization of feature importance metrics

### Model Training & Evaluation
- Support for both classification and regression tasks
- Cross-validation for robust performance estimation
- Comprehensive evaluation metrics

### Supported Models
- **Classification**: Decision Trees, Random Forests, K-Nearest Neighbors, Naive Bayes, Support Vector Machines, Neural Networks
- **Regression**: Decision Trees, Random Forests, K-Nearest Neighbors, Support Vector Machines, Neural Networks, Linear Regression

### Visualization
- Automated generation of performance visualizations
- Feature importance plots
- Confusion matrices for classification tasks
- Actual vs. Predicted plots for regression tasks
- Model comparison charts

## Requirements
- Python 3.x
- Required libraries:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ml-model-comparison.git
cd ml-model-comparison

# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
# Import the necessary modules
from ml_framework import load_data, main

# Run the full pipeline with default settings
main()
```

### Custom Usage
```python
# Import specific components
from ml_framework import (
    load_data,
    remove_highly_correlated_features,
    compare_feature_selection_methods,
    detect_outliers,
    scale_data,
    prepare_classification_data,
    train_test_split_data,
    run_decision_tree,
    compare_models
)

# Load and prepare your data
data = load_data('your_dataset.csv')
X = data.drop(columns=['target_column'])
y = data['target_column']

# Select specific features
features = ['feature1', 'feature2', 'feature3', 'feature4']
X = X[features]

# Process data
X_clean = detect_outliers(X, features, method='iqr', clean_outliers=True)
X_scaled, _ = scale_data(X_clean, features)

# Split data
X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

# Train and evaluate a specific model
model, results = run_decision_tree(X_train, X_test, y_train, y_test, 'regression')
```

## Output
The framework automatically creates a `results` directory containing:
- Model performance metrics in CSV format
- Feature selection results
- Visualization plots for all models
- Comparison charts

## Advanced Configuration
The framework supports advanced configuration through parameters in the various functions:
- Feature selection thresholds
- Outlier detection parameters
- Model hyperparameters
- Cross-validation folds
- Evaluation metrics

## Example Results
After running the pipeline, you'll get comprehensive model comparison results like:

**Classification Models:**
- Accuracy scores across all models
- Precision, recall, and F1 scores
- Confusion matrices
- Feature importance rankings

**Regression Models:**
- Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² scores
- Feature coefficient analysis

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The scikit-learn team for their excellent machine learning library
- All contributors and users of this framework

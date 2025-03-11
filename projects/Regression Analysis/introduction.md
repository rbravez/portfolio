# Regression Analysis Run Book

Regression analysis is a fundamental statistical and machine learning technique used to model relationships between variables. It helps us understand how one or more independent variables (predictors) influence a dependent variable (outcome). 

## Types of Regression

There are different types of regression models, each suited for specific types of data and relationships.

1. Linear Regression: Models a straight-line relationship between variables.
2. Multiple Regression: Extends linear regression to include multiple predictors.
3. Polynomial Regression: Captures non-linear relationships.
4. Logistic Regression: Used for classification problems where the outcome is binary (e.g., yes/no, fraud/no fraud).
5. Regularized Regression (Ridge/Lasso): Helps in cases of multicollinearity and feature selection.

## About this Project
This project provides a Python-based regression analysis toolkit, offering functions for data preprocessing, model fitting, evaluation, and visualization. Here is an example on 
how to graoh the ROC curve, given a model. The tool-kit can be found in the file called `functions.py`. 

```ruby
def curve_roc(x_test_modelo, y_test, modelo):
    """
    Calculates ROC curve and shows the graph.

    Parameters:
        x_test_modelo (array-like): Test data.
        y_test (array-like): Target test data.
        modelo (dict): Dictionary that contains the model.

    Returns:
        float: Area below the curve.
    """
    # Calculates the probabilities for the defined class
    y_prob = modelo['Modelo'].predict_proba(x_test_modelo)[:, 1]
    
    # Calculates false positives and true positives
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Graph of the ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curve ROC')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positives (FPR)')
    plt.ylabel('True Positives (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()
    
    # Calculates the area below the curve (AUC)
    AUC = roc_auc_score(y_test, y_prob)
    print('Area below the curve =', AUC)
    
    return AUC
);
```
## Usage Example
TBD

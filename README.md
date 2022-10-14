# binary-classification-with-SHAP-explanation
## Summary

This is the basic machine learning training process, using common binary classification models to predict the probability of muscle loss in cancer patients during postoperative treatment. Use bootstraps for validation. Trained the models with default parameters, and you can use GridSearch to find the suitable model parameters for the data. Combine or separate oversampling and Undersampling methods for imbalanced data. Common metrics and ROC curve to observe the prediction results. SHAP (SHapley Additive exPlanations) method to observe the contribution of features to prediction.

Brief introduction of data: 

__metrics:__
- precision: 
- sensitivity (recall):
- specificity:
- F1 score: 
- accuracy: 
- AUC


__models:__
- LogisticRegression
- Random Forest
- K-Nearest Neighbor
- Bernoulli Naive-Bayes
- Light Gradient Boosting
- Support Vector Classifier 
- XGBoost
- Ensemble model (LR + SVC)

__sources:__  
- SHAP source: https://github.com/slundberg/shap  
- GridSearch source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

## Environment and main packages

Anaconda 2019.07 for 64-bit (Python 3.7.3)  
(Anaconda Old package lists: https://docs.anaconda.com/anaconda/packages/oldpkglists/ )  

__main packages:__
- scikit-learn 1.0.1  
- imbalanced-learn 0.9.0  
- shap 0.40.0  
- matplotlib 3.3.0  

## Code
- Training (with bootstraps)
  - Randomly cut training and validation datasets and repeat 500 times.
  - Get model metrics and select the model.  ..
  
Result:  ..

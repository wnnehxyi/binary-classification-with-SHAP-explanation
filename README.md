# binary-classification-with-SHAP-explanation
## Summary

This is the basic machine learning training process, using common binary classification models to predict the probability of muscle loss in cancer patients during postoperative treatment. Use bootstraps for validation. Trained the models with default parameters, and you can use GridSearchCV to find the suitable model parameters for the data. Combine or separate oversampling and Undersampling methods for imbalanced data. Common metrics and ROC curve to observe the prediction results. SHAP (SHapley Additive exPlanations) method to observe the contribution of features to prediction.

Brief introduction of data: Ovarian cancer patients from a certain hospital with clinical characteristics such as age, malignant ascites, BMI, laboratory data...etc. The dataset was used to predict whether a patient would have muscle loss after treatment based on these clinical features. The dataset is not provided due to confidentiality.

__metrics:__
- precision: TP/TP+FP
- sensitivity (recall): TP/TP+FN
- specificity: TN/TN+FP
- F1 score: 2*(precision*recall/precision+recall)
- accuracy: TP+TN/TP+TN+FP+FN
- AUC: Area Under the ROC Curve  

__TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative__  
__Receiver operating characteristic (ROC) curve: by plotting the true positive rate (TPR) against the false positive rate (FPR)__

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
- GridSearchCV source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

## Environment and main packages

Anaconda 2019.07 for 64-bit (Python 3.7.3)  
(Anaconda Old package lists: https://docs.anaconda.com/anaconda/packages/oldpkglists/)  

__main packages:__
- scikit-learn 1.0.1  
- imbalanced-learn 0.9.0  
- shap 0.40.0  
- matplotlib 3.3.0  

## Code
- Training (with bootstraps)
  - Randomly cut training and validation datasets and repeat 500 times.
  - Get model metrics and select the model algorithms.  

- Predict independent data
  - Evaluate the selected model performance using the independent test dataset.
  
- GridSearchCV in bootstraps
  - Use GridSearchCV to find the best candidate hyperparameters in each bootstrap.
  
- ROC curve
  - Plot the ROC curve in bootstraps.
  
- CI and DES
  - Calculate the confidence interval of each metric in the bootstraps and the descriptive statistics of each prediction model.
  
- SHAP in bootstraps
  - Calculate the feature contribution in each bootstrap.
  
- SHAP visualization
  - Visualize feature contribution
    - SHAP summary plot
    - SHAP dependence plot
    - SHAP force plot

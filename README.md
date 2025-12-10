# nithyhaprabharp-uptor_112_final_project-supervised_learning-Decision_tree_CV_hyperparameter-tune
Drug Classification with Decision Tree

***Project Description:*** The project is to classify and predict the drugs type that might be suitable for the patient.

***Goal***The goal of the project is to train the ***Decision Tree Machine learning model*** to classify and predict the drugs type that might be suitable for the patient based on the features/attributes.

In this assignment,I applied a supervised machine learning pipeline to classify patients into appropriate drug categories based on medical attributes such as Age, Sex, Blood Pressure, Cholesterol, and Na_to_K ratio.

-The chosen model was a Decision Tree Classifier, trained with cross-validation and hyperparameter tuning (GridSearchCV) to ensure robust performance and avoid overfitting.

-DECISION TREE were selected for their interpretability, making it easy to explain medical decisions through clear rules and splits.

***Decision Tree***

- Interpretability: Easy to explain with clear rules (“If Na_to_K > X → DrugY”).
- Single model: Relies on one tree, so it can capture patterns but is prone to overfitting.
- Transparency: Stakeholders can see exactly how decisions are made.

***CROSS-VALIDATION***

It’s a systematic evaluation method that repeatedly splits the dataset into training and testing folds to check how well the model generalizes.

***HYPER PARAMETER TUNING***

Hyperparameter Tuning = Fine-Tuning the Model’s Learning Capacity

***Key Steps:***

- Data Preparation → Features: Age, Sex, BP, Cholesterol, Na_to_K  Target: Drug
- Train/Test Split → 80% training, 20% testing
- Hyperparameter Tuning → GridSearchCV explored:
- criterion: gini, entropy
- max_depth: 3, 5, 10, None
- min_samples_split: 2, 5, 10

***Outcome Analysis of Best CV score & Test accuracy***

- Best CV Score: 99.4%
- Test Accuracy: 97.5%

- Model is highly accurate and interpretable.
- Na_to_K emerged as the most important feature, followed by BP, Age, and Cholesterol
- Decision Tree visualization clearly shows the rules behind drug prescriptions


***conclusion***

The Drug Classification with Decision Tree project showed that patient attributes like age, sex, blood pressure, cholesterol, and sodium‑to‑potassium ratio can be used to reliably predict suitable drug types.
With hyperparameter tuning and cross‑validation, the model achieved solid accuracy while remaining interpretable, making it a practical baseline for healthcare decision support.

***Decision Tree*** modeling proved effective in classifying patient attributes to predict suitable drug types with accuracy and interpretability.



***Assignment Files***

- [ML_Assignment_4_supervised_decisiontree_hyperparameter_tuning_drug200.py](ML_Assignment_4_supervised_decisiontree_hyperparameter_tuning_drug200.py)  
- [ML_Assignment_4_supervised_decisiontree_hyperparameter_tuning_drug200.ipynb](ML_Assignment_4_supervised_decisiontree_hyperparameter_tuning_drug200.ipynb)


***Data***

- [drug200.csv](drug200.csv)


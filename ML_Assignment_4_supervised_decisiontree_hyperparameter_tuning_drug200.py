
"""Drug200 Decision Tree Pipeline"""

print("-----ASSIGNMENT 4 - HYPER PARAMETER TUNING & CROSS VALIDATION-----")
print("-----MODEL - DECISION TREE -----")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# """1. Load dataset"""

df = pd.read_csv("drug200.csv")


# """ 2. Data Preprocessing"""

"""dropping column drug from x-axis"""

X = df.drop("Drug", axis=1)
y = df["Drug"]

# """Encode categorical features"""
categorical_cols = ["Sex", "BP", "Cholesterol"]
encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])



# """ 3. Train/Test split """

"""training on 80% of data and test on 20% data from the dataset"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



#  4. Define Decision Tree + Hyperparameter Grid

# """ Hyperparameter Grid """


param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}

# """tuning the Decision Tree model via GridSearchCV."""

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,              # 5-fold cross-validation
    scoring="accuracy"
)



# 5. Fit model

grid.fit(X_train, y_train)



#  6. Model Evaluate

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nBest Parameters:\n", grid.best_params_)

print("\nBest CV Score:\n", grid.best_score_)

print("\nTest Accuracy:\n", accuracy_score(y_test, y_pred))

print("==============================================")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("============================")


importances = best_model.feature_importances_


# Create a DataFrame for feature importances
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

#  To display numeric score- weightage each feature carried in the overall prediction process.

print("\nFeature Importance Scores:")
print(feat_imp.sort_values(by="Importance", ascending=False))




# 7. Visualizations


# Bar chart: distribution of drug types
plt.figure(figsize=(8,5))
sns.countplot(x="Drug", hue="BP", data=df, palette="Set2")
# sns.countplot(x="Drug", hue="Cholesterol", data=df, palette="Set2")
plt.title("Distribution of Drug Types")
plt.xlabel("Drug")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# Pie chart: proportion of drug types

plt.figure(figsize=(6,6))
df["Drug"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("Set3"))
plt.title("Proportion of Drug Types")
plt.ylabel("")
plt.show()



# Scatterplot: Age vs Na_to_K colored by Drug

plt.figure(figsize=(8,6))
sns.scatterplot(x="Age", y="Na_to_K", hue="Drug", data=df, palette="Set1")
plt.title("Age vs Na_to_K by Drug Type")
plt.xlabel("Age")
plt.ylabel("Na_to_K")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# Visualize the best decision tree

plt.figure(figsize=(20,10))
plot_tree(
    best_model,                # your tuned tree
    feature_names=X.columns,   # show feature names
    class_names=best_model.classes_, # show drug labels
    filled=True,               # color nodes by class
    rounded=True,              # rounded boxes for readability
    fontsize=10
)
plt.title("Decision Tree Visualization (Best Model)")
plt.show()


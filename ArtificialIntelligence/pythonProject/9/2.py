import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load breast cancer dataset
data = load_breast_cancer()
# print(type(data))
X = data.data
y = data.target
# print(type(X))
df = pd.DataFrame(data.data, columns=data.feature_names)
# Output sample feature information
print("Sample Features Information:")
print(df.info)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate three types of Naive Bayes models
models = {
    "BernoulliNB": BernoulliNB(),
    "MultinomialNB": MultinomialNB(),
    "GaussianNB": GaussianNB()
}

for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    # Output feature importance for GaussianNB
    if name == "GaussianNB":
        print("Feature Means:")
        print(model.theta_)

    # For BernoulliNB and MultinomialNB, as they are not probability-based,
    # we can't directly get feature importance in the same way as GaussianNB.
    # However, we can output the learned parameters for inspection.
    elif name == "BernoulliNB":
        print("Feature Log Probability (BernoulliNB):")
        print(model.feature_log_prob_)

    elif name == "MultinomialNB":
        print("Feature Log Probability (MultinomialNB):")
        print(model.feature_log_prob_)

# Randomly test one sample
random_index = np.random.randint(0, len(X_test))
sample = X_test[random_index]
true_label = y_test[random_index]

print("\nRandom Sample:")
print("True Label:", true_label)

# Output predicted probabilities for each class for each model
for name, model in models.items():
    prob = model.predict_proba([sample])[0]
    print(f"{name} Predicted Value:")
    print(model.predict([sample]))
    print(f"{name} Predicted Probabilities:")
    for i, p in enumerate(prob):
        print(f"Class {i}: {p}")

# src/model.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier

# -----------------------------
# Logistic Regression
# -----------------------------
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        penalty="l1"
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Support Vector Machine
# -----------------------------
def train_svm(X_train, y_train):
    model = SVC(
        class_weight="balanced",
        probability=True
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Linear SVM (Faster Alternative)
# -----------------------------
def train_linear_svm(X_train, y_train):
    model = LinearSVC(class_weight="balanced")
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Bagging (Optional)
# -----------------------------
def train_bagging(base_model, X_train, y_train, n_estimators=10):
    bagging_model = BaggingClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        bootstrap=True,
        n_jobs=-1
    )
    bagging_model.fit(X_train, y_train)
    return bagging_model


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, report
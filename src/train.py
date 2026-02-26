# src/train.py

from preprocessing import load_data, split_data, vectorize_text
from model import (
    train_logistic_regression,
    train_svm,
    evaluate_model
)

# -----------------------------
# Main Training Pipeline
# -----------------------------
def main():

    # Load Dataset
    df = load_data("../data/fake_job_postings.csv")

    # Split Data
    X_train, X_test, y_train, y_test = split_data(
        df,
        text_column="job_description",
        target_column="fraudulent"
    )

    # TF-IDF Vectorization
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(
        X_train,
        X_test
    )

    # -------------------------
    # Logistic Regression
    # -------------------------
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_tfidf, y_train)

    lr_accuracy, lr_report = evaluate_model(
        lr_model,
        X_test_tfidf,
        y_test
    )

    print("\nLogistic Regression Results")
    print("Accuracy:", lr_accuracy)
    print(lr_report)

    # -------------------------
    # Support Vector Machine
    # -------------------------
    print("Training SVM...")
    svm_model = train_svm(X_train_tfidf, y_train)

    svm_accuracy, svm_report = evaluate_model(
        svm_model,
        X_test_tfidf,
        y_test
    )

    print("\nSVM Results")
    print("Accuracy:", svm_accuracy)
    print(svm_report)


if __name__ == "__main__":
    main()
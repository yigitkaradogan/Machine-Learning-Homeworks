import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Loading and Preprocessing
def preprocess_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['data'])

    features = df['data'].iloc[0].split('\t')  # Extract feature names from the first row
    df = df.iloc[1:]  # Remove the header row

    # Split each data row into features
    X = df['data'].str.split('\t', expand=True).astype(float)
    X.columns = features

    y = X.pop('Class')  # Extract the target class

    return X, y

X_train, y_train = preprocess_data("Spambase_train.csv")
X_val, y_val = preprocess_data("Spambase_val.csv")
X_test, y_test = preprocess_data("Spambase_test.csv")

X_train.hist(figsize=(15, 10))
plt.xlabel("Feature Value")
plt.ylabel("Number of Samples")
plt.title("Distribution of Features (X_train)")
plt.tight_layout()
plt.show()

def handle_skew(data):
  skewness = data.skew()

  if skewness > 0.5:
    return np.log1p(data)
  else:
    return data

for column in X_train.columns:
    X_train[column] = handle_skew(X_train[column])

X_train.hist(figsize=(15, 10))
plt.xlabel("Feature Value")
plt.ylabel("Number of Samples")
plt.title("Distribution of Features (X_train)")
plt.tight_layout()
plt.show()

# 2. Logistic Regression
def sigmoid(z):
    z = np.clip(z, -700, 1000)
    return 1 / (1 + np.exp(-z))

def predict(X, weights):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return (predictions >= 0.8).astype(int)

def log_likelihood(y, z):
    return np.sum(y * sigmoid(z) + (1-y) * (1 - sigmoid(z)))

# 3. Gradient Ascent
def gradient_ascent(X, y, lr, iterations):
    m, n = X.shape
    weights = np.zeros(n)
    log_likelihoods = []

    for i in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        error = y - predictions

        # Calculate Log-Likelihood
        loglikelihood = log_likelihood(y, np.dot(X, weights))
        log_likelihoods.append(loglikelihood)

        if i % 100 == 0:  
            print(f"Iteration {i+1}: log-likelihood = {loglikelihood:.4f}")

        gradient = np.dot(X.T, error) / m
        weights += lr * gradient
    return weights, log_likelihoods

# 4. Hyperparameter Tuning
def confusion_matrix(y_true, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == predicted_label and true_label == 1:
            tp += 1
        elif true_label == predicted_label and true_label == 0:
            tn += 1
        elif true_label != predicted_label and true_label == 0:
            fp += 1
        else:  # true_label != predicted_label and true_label == 1
            fn += 1

    return tp, fp, tn, fn

learning_rates = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
best_f1 = 0
best_lr = None
f1_values = []

for lr in learning_rates:
    weights, log_likelihoods = gradient_ascent(X_train, y_train, lr, 1000)
    plt.plot(range(1, 1001), log_likelihoods, label=f"lr={lr}")

    predictions = predict(X_val, weights)
    tp, fp, tn, fn = confusion_matrix(y_val, predictions)
    print("Confusion Matrix:\n", np.array([[tp, fp], [fn, tn]]))
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    f1_values.append(f1)

    if f1 > best_f1:
        best_f1 = f1
        best_lr = lr

plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Iteration for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(learning_rates, f1_values, marker='o')
plt.xscale('log')
plt.xlabel('Learning Rate (Log Scale)')
plt.ylabel('F1 Score')
plt.title('F1 Scores vs. Learning Rates')
plt.grid(True)
plt.show()

# 5. Evaluation and Reporting
weights, log_likelihoods = gradient_ascent(X_train, y_train, best_lr, 1000)
predictions = predict(X_test, weights)

# Confusion Matrix
tp, fp, tn, fn = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", np.array([[tp, fp], [fn, tn]]))

def calculate_metrics(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    if tn + fn == 0:
        npv = 0
    else:
        npv = tn / (tn + fn)

    if fp + tn == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    if fp + tp == 0:
        fdr = 0
    else:
        fdr = fp / (fp + tp)

    if (4 * precision + recall) == 0:
        f2 = 0
    else:
        f2 = (5 * precision * recall) / (4 * precision + recall)

    return accuracy, precision, recall, f1, npv, fpr, fdr, f2

# Calculate metrics
accuracy, precision, recall, f1, npv, fpr, fdr, f2 = calculate_metrics(y_test, predictions)

# Print metrics
print("Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"F2 Score: {f2:.2f}")
print(f"Negative Predictive Value (NPV): {npv:.2f}")
print(f"False Positive Rate (FPR): {fpr:.2f}")
print(f"False Discovery Rate (FDR): {fdr:.2f}")

#Q3-B
X_train, y_train = preprocess_data("Spambase_train.csv")
X_val, y_val = preprocess_data("Spambase_val.csv")
X_test, y_test = preprocess_data("Spambase_test.csv")
def predict1(X, weights):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return (predictions >= 0.32).astype(int)

def predict2(X, weights):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return (predictions >= 0.12).astype(int)

def mini_batch_gradient_ascent(X, y, lr, iterations, batch_size=32):
    m, n = X.shape
    weights = np.zeros(n)

    for _ in range(iterations):
        # Shuffle for more "randomness"
        data = np.column_stack((X, y))
        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1]

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)  # Handle last batch size
            X_batch, y_batch = X[start:end], y[start:end]

            predictions = sigmoid(np.dot(X_batch, weights))
            error = y_batch - predictions

            gradient = np.dot(X_batch.T, error) / batch_size
            weights += lr * gradient

    return weights

def stochastic_gradient_ascent(X, y, lr, iterations):
    m, n = X.shape
    weights = np.zeros(n)
    data = np.column_stack((X, y))

    for _ in range(iterations):
        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1]

        for i in range(m):  # Iterate over individual samples
            xi = X[i:i+1]  # Reshape as a row vector
            yi = y[i:i+1]

            predictions = sigmoid(np.dot(xi, weights))
            error = yi - predictions

            gradient = np.dot(xi.T, error)
            weights += lr * gradient

    return weights

# Mini-Batch Gradient Ascent
weights_mb = mini_batch_gradient_ascent(X_train, y_train, best_lr, 1000)
predictions_mb = predict1(X_test, weights_mb)

# Calculate metrics for mb
accuracy, precision, recall, f1, npv, fpr, fdr, f2 = calculate_metrics(y_test, predictions_mb)

# Print metrics nicely formatted
tp, fp, tn, fn = confusion_matrix(y_test, predictions_mb)
print("Confusion Matrix for Mini-Batch Gradient Ascent:\n", np.array([[tp, fp], [fn, tn]]))
print("Performance Metrics for Mini-Batch Gradient Ascent:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"F2 Score: {f2:.2f}")
print(f"Negative Predictive Value (NPV): {npv:.2f}")
print(f"False Positive Rate (FPR): {fpr:.2f}")
print(f"False Discovery Rate (FDR): {fdr:.2f}")

# Stochastic Gradient Ascent
weights_sgd = stochastic_gradient_ascent(X_train, y_train, best_lr, 1000)
predictions_sgd = predict2(X_test, weights_sgd)

# Calculate metrics for sga
accuracy, precision, recall, f1, npv, fpr, fdr, f2 = calculate_metrics(y_test, predictions_sgd)

# Print metrics for sga
tp, fp, tn, fn = confusion_matrix(y_test, predictions_sgd)
print("Confusion Matrix for Stochastic Gradient Ascent:\n", np.array([[tp, fp], [fn, tn]]))
print("Performance Metrics for Stochastic Gradient Ascent:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"F2 Score: {f2:.2f}")
print(f"Negative Predictive Value (NPV): {npv:.2f}")
print(f"False Positive Rate (FPR): {fpr:.2f}")
print(f"False Discovery Rate (FDR): {fdr:.2f}")
import numpy as np


def metrics(y, y_pred):
    """
    y: (B, T) INT - True labels
    y_pred: (B, T, C) FLOAT - Predicted probabilities

    Returns:
    - Perplexity
    - F1 Score
    - Precision
    - Recall
    - Cross Entropy
    """

    # Make sure y_pred is between 0 and 1
    if not (np.all(y_pred >= 0) and np.all(y_pred <= 1)):
        # Softmax
        y_pred = np.exp(y_pred) / np.sum(
            np.exp(y_pred), axis=-1, keepdims=True
        )

    assert np.all(y_pred >= 0) and np.all(
        y_pred <= 1
    ), "y_pred must be between 0 and 1"

    # Add a small epsilon for numerical stability
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Cross Entropy
    y_one_hot = np.eye(y_pred.shape[-1])[y]
    cross_entropy = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=-1))

    # Perplexity
    perplexity = 2**cross_entropy

    # Predicted classes
    y_pred_class = np.argmax(y_pred, axis=-1)

    # True Positives, False Positives, and False Negatives
    TP = np.sum(y == y_pred_class)
    FP = np.sum(y != y_pred_class)
    FN = FP  # binary setup, false positives and false negatives are equivalent

    # Precision, Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall)

    return perplexity, f1, precision, recall, cross_entropy

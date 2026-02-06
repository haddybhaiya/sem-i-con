# evaluation/metrics.py

from sklearn.metrics import classification_report, accuracy_score

def compute_metrics(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)

    class_accuracy = {}
    for idx, cls in enumerate(class_names):
        total = sum(1 for y in y_true if y == idx)
        correct = sum(
            1 for yt, yp in zip(y_true, y_pred)
            if yt == idx and yp == idx
        )
        class_accuracy[cls] = round(correct / total, 3) if total > 0 else 0.0

    return accuracy, class_accuracy, report

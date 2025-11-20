import numpy as np
from models import (
    SimplePredictor,
    SVMClassifier,
    RFClassifier,
    KNNClassifier,
    MLPClassifier,
    NBClassifier   
)
from data_prep import RecommenderDataPrep, MODEL_FEATURE_COLS
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

def plot_roc_curves(y_true, scores_dict, out):
    fig = go.Figure()
    for name, y_score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Chance", line=dict(dash="dash")))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend_title_text=None,
        width=900, height=550
    )
    fig.write_image(out)


def plot_pr_curves(y_true, scores_dict, out):
    fig = go.Figure()
    pos_rate = (sum(y_true) / len(y_true)) if len(y_true) else 0.0
    for name, y_score in scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                 name=f"{name} (AP={ap:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[pos_rate, pos_rate], mode="lines",
                             name=f"Baseline (pos rate={pos_rate:.3f})",
                             line=dict(dash="dash")))
    fig.update_layout(
        title="Precision-Recall Curve Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        legend_title_text=None,
        width=900, height=550
    )
    fig.write_image(out)


def print_report(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def main():
    print("Loading and preparing data...")
    data_prep = RecommenderDataPrep(True)
    data_prep.load_and_prepare(False)
    
    models = {
        "Simple": (SimplePredictor(), MODEL_FEATURE_COLS),
        "SVM": (SVMClassifier(), MODEL_FEATURE_COLS),
        "RandomForest": (RFClassifier(), MODEL_FEATURE_COLS),
        "KNN": (KNNClassifier(), MODEL_FEATURE_COLS),
        "MLP": (MLPClassifier(), MODEL_FEATURE_COLS),
        "NB": (NBClassifier(), MODEL_FEATURE_COLS),
    }

    for name, (model, feature_cols) in models.items():
        print(f"Training {name}...")
        X_train, y_train = data_prep.get_training_data(feature_cols)
        model.fit(X_train, y_train)

    predictions = {}
    probas = {}

    for name, (model, feature_cols) in models.items():
        print(f"Generating recommendations for {name}...")
        X_test = data_prep.get_test_data(feature_cols)
        predictions[name] = model.predict(X_test)
        probas[name] = model.predict_proba(X_test)

    y_test = data_prep.get_test_results()

    for name in models.keys():
        print_report(name, y_test, predictions[name], probas[name])

    plot_roc_curves(y_test, probas, out="out/roc_curve.png")
    plot_pr_curves(y_test, probas, out="out/pr_curve.png")


if __name__ == "__main__":
    main()
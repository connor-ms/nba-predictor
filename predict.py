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

def main():
    print("Loading and preparing data...")
    data_prep = RecommenderDataPrep(False)
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

    # Todo


if __name__ == "__main__":
    main()
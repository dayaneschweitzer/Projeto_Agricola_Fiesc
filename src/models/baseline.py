from typing import Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_baseline_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Treina um modelo baseline de classificação (RandomForest)
    para distinguir as 5 classes de maçãs a partir dos espectros.

    Retorna um dicionário com:
        - "model": modelo treinado
        - "accuracy": acurácia no conjunto de teste
        - "report": classification_report (string)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    return {
        "model": clf,
        "accuracy": acc,
        "report": report,
    }

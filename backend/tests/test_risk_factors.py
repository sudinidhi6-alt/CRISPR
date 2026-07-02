import importlib
import sys
import types


class ConstantModel:
    def __init__(self, *args, **kwargs):
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in X]


def install_dummy_sklearn():
    sklearn = types.ModuleType("sklearn")
    ensemble = types.ModuleType("sklearn.ensemble")
    linear_model = types.ModuleType("sklearn.linear_model")
    neural_network = types.ModuleType("sklearn.neural_network")
    model_selection = types.ModuleType("sklearn.model_selection")
    metrics = types.ModuleType("sklearn.metrics")

    ensemble.RandomForestClassifier = ConstantModel
    linear_model.LogisticRegression = ConstantModel
    neural_network.MLPClassifier = ConstantModel
    model_selection.train_test_split = lambda X, y, test_size=0.3, random_state=None, stratify=None: (X, X, y, y)
    metrics.confusion_matrix = lambda y_true, y_pred, labels=None: [[0, 0], [0, 0]]
    metrics.precision_score = lambda y_true, y_pred, zero_division=0: 0.0
    metrics.recall_score = lambda y_true, y_pred, zero_division=0: 0.0
    metrics.f1_score = lambda y_true, y_pred, zero_division=0: 0.0
    metrics.roc_curve = lambda y_true, y_score: ([0.0, 1.0], [0.0, 1.0], None)
    metrics.auc = lambda fpr, tpr: 1.0

    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.ensemble"] = ensemble
    sys.modules["sklearn.linear_model"] = linear_model
    sys.modules["sklearn.neural_network"] = neural_network
    sys.modules["sklearn.model_selection"] = model_selection
    sys.modules["sklearn.metrics"] = metrics


install_dummy_sklearn()
sys.path.insert(0, "/Users/dhiya/Desktop/CRISPR/backend")
app_module = importlib.import_module("app")


def test_risk_factor_importances_change_with_input():
    engine = app_module.CRISPRMLPipeline()

    first = engine.risk_factor_breakdown(
        "GCCAATCGATCGATCGATCG",
        "GCCAATCGATCGATCGATGG",
    )
    second = engine.risk_factor_breakdown(
        "GCCAATCGATCGATCGATCG",
        "AAAAAAAAAAAAAAAAAAAA",
    )

    first_importance = {item["factor"]: item["importance"] for item in first["factors"]}
    second_importance = {item["factor"]: item["importance"] for item in second["factors"]}

    assert first_importance != second_importance
    assert abs(sum(first_importance.values()) - 100.0) < 1e-9
    assert abs(sum(second_importance.values()) - 100.0) < 1e-9

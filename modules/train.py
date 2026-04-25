
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_all_models(X, y):

    is_classification = len(set(y)) < 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    results = {}
    best_model = None
    best_score = -999

    if is_classification:
        models = {
            "LogReg": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier()
        }
    else:
        models = {
            "Linear": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor()
        }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if is_classification:
            score = accuracy_score(y_test, preds)
        else:
            score = r2_score(y_test, preds)

        results[name] = round(score, 4)

        if score > best_score:
            best_score = score
            best_model = model

    return results, best_model

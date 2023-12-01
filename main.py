import time
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFE


def train_rf_with_random_search(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df['Severity'],
                                                        test_size=0.3,
                                                        stratify=df['Severity'],
                                                        random_state=42)
    clf = RandomForestClassifier()
    new_clf = RandomForestClassifier()

    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    criterion = ['gini', 'entropy', 'log_loss']
    max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
    max_depth.append(None)
    min_samples_leaf = [1, 2, 4]
    min_samples_split = [5, 10, 15]
    random_grid = {'n_estimators': n_estimators,
                   'criterion': criterion,
                   'max_depth': max_depth,
                   'min_samples_leaf': min_samples_leaf,
                   'min_samples_split': min_samples_split}
    random_search = RandomizedSearchCV(estimator=new_clf,
                                       param_distributions=random_grid,
                                       n_iter=20,
                                       scoring='accuracy',
                                       cv=5,
                                       verbose=1,
                                       random_state=42,
                                       n_jobs=-1)

    start_time = time.time()
    clf.fit(X_train, y_train)
    random_search.fit(X_train, y_train)
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)
    training_time = time.time() - start_time

    rf_pred = clf.predict(X_test)
    best_rf_pred = best_rf.predict(X_test)

    print(random_search.best_params_)

    print("\nTraining time: ", training_time)
    print("\nTest accuracy: ")
    print("Tuned model: ", accuracy_score(y_test, best_rf_pred))
    print("Default model: ", accuracy_score(y_test, rf_pred))

    print("\nClassification report: ")
    print(classification_report(y_test, best_rf_pred))


def train_knn_with_random_search(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df['Severity'],
                                                        test_size=0.3,
                                                        stratify=df['Severity'],
                                                        random_state=42)
    clf = KNeighborsClassifier()
    new_clf = KNeighborsClassifier()

    n_neighbors = [3, 4, 5, 6, 7, 8, 9, 10]
    weights = ['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size = [int(x) for x in np.linspace(10, 100, num=10)]
    p = [1, 2]
    random_grid = {'n_neighbors': n_neighbors,
                   'weights': weights,
                   'algorithm': algorithm,
                   'leaf_size': leaf_size,
                   'p': p}
    random_search = RandomizedSearchCV(estimator=new_clf,
                                       param_distributions=random_grid,
                                       n_iter=20,
                                       scoring='accuracy',
                                       cv=5,
                                       verbose=1,
                                       random_state=42,
                                       n_jobs=-1)

    start_time = time.time()
    clf.fit(X_train, y_train)
    random_search.fit(X_train, y_train)
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)
    training_time = time.time() - start_time

    rf_pred = clf.predict(X_test)
    best_rf_pred = best_rf.predict(X_test)

    print(random_search.best_params_)

    print("\nTraining time: ", training_time)
    print("\nTest accuracy: ")
    print("Tuned model: ", accuracy_score(y_test, best_rf_pred))
    print("Default model: ", accuracy_score(y_test, rf_pred))

    print("\nClassification report: ")
    print(classification_report(y_test, best_rf_pred))


def train_rf_with_feature_selection(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df['Severity'],
                                                        test_size=0.3,
                                                        stratify=df['Severity'],
                                                        random_state=42)

    clf = RandomForestClassifier()
    selector = RFE(clf, n_features_to_select=20, verbose=2)  # tried 10, 20, 30, 40, 50, and 60

    print("Training the model...")
    start_time = time.time()
    selector = selector.fit(X_train, y_train)
    clf.fit(selector.transform(X_train), y_train)
    training_time = time.time() - start_time

    rf_pred = clf.predict(selector.transform(X_test))
    print("\nTraining time: ", training_time)
    print("\nTest accuracy: ")
    print("Tuned model: ", accuracy_score(y_test, rf_pred))

    print("\nClassification report: ")
    print(classification_report(y_test, rf_pred))


def train_sgd(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df['Severity'],
                                                        test_size=0.3,
                                                        stratify=df['Severity'],
                                                        random_state=42)
    clf = SGDClassifier()
    new_clf = SGDClassifier()

    alpha = [0.001, 0.01]
    max_iter = [1000, 2000, 3000]
    # tol = [1e-3, 1e-4]
    loss = ['hinge', 'log_loss', 'modified_huber']
    penalty = ['l2', 'l1', 'elasticnet']

    param_grid = {'alpha': alpha,
                  'max_iter': max_iter,
                  # 'tol': tol,
                  'loss': loss,
                  'penalty': penalty}
    grid_search = GridSearchCV(estimator=new_clf,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=5,
                               verbose=1,
                               n_jobs=-1)

    start_time = time.time()
    clf.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)
    training_time = time.time() - start_time

    rf_pred = clf.predict(X_test)
    best_rf_pred = best_rf.predict(X_test)

    print(grid_search.best_params_)

    print("\nTraining time: ", training_time)
    print("\nTest accuracy: ")
    print("Tuned model: ", accuracy_score(y_test, best_rf_pred))
    print("Default model: ", accuracy_score(y_test, rf_pred))

    print("\nClassification report: ")
    print(classification_report(y_test, best_rf_pred))


def run(df):
    while True:
        print("Please select the algorithm:"
              "\n1. Random Forest"
              "\n2. Stochastic Gradient Descent"
              "\n3. K-Nearest Neighbors"
              "\n4. Exit")
        approach = input("Enter command: ").strip()
        if not approach:
            continue
        if approach == "1":
            print("Please select the tuning method:"
                  "\n1. RandomSearchCV"
                  "\n2. Feature Selection with sklearn RFE"
                  "\n3. Exit")
            method = input("Enter command: ").strip()
            if not method:
                continue
            elif method == "1":
                train_rf_with_random_search(df)
            elif method == "2":
                train_rf_with_feature_selection(df)
            elif method == "3":
                break
            else:
                print("Invalid command!")
        elif approach == "2":
            print("Training SGD with Grid Search...")
            train_sgd(df)
        elif approach == "3":
            print("Training KNN with Random Search...")
            train_knn_with_random_search(df)
        elif approach == "4":
            break
        else:
            print("Invalid command!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_pickle('tfidf_processed_dataframe.pkl')
    # df = pd.read_pickle('distillbert_processed_dataframe.pkl')
    # train_rf_with_feature_selection(df)
    run(df)

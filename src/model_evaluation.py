from config import Config
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix



def model_evaluation(models,X_train,X_test,y_train,y_test):
    for model_name, model in models.items():
        model.fit(X_train, y_train)  # Seen Data
    y_pred = model.predict(X_test)   # Unseen Data
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

    return classification_report,confusion_matrix
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Iris dataset
def dataload():
    X, y = datasets.load_iris(return_X_y=True)
    return X,y

# Split the data into training and test sets
def split(x,y):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42
)
    return X_train, X_test, y_train, y_test

def model_training(X_train,y_train,X_test,params):
   
# Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

# Predict on the test set
    y_pred = lr.predict(X_test)
    return lr,y_pred

def Scores(y_test,y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    ps=precision_score(y_test,y_pred,average="weighted")
    rs=recall_score(y_test,y_pred,average="weighted")
    # f1s=f1_score(y_test,y_pred)

    return accuracy,ps,rs


def pipeline():
    
    params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

    x,y=dataload()
    xtrain,xtest,ytrain,ytest=split(x,y)
    lr,pred=model_training(xtrain,ytrain,xtest,params)
    a,p,r=Scores(ytest,pred)
    print(a,p,r)

    # mlflow.set_experiment("MLflow LifeCycle")


    with mlflow.start_run():
        mlflow.log_params(params)

        mlflow.log_metric("accuracy",a)
        mlflow.log_metric("presicion",a)
        mlflow.log_metric("recall",r)

        remote_url="https://dagshub.com/SKrishna-7/complete-mlflow-lifecycle.mlflow"
        mlflow.set_tracking_uri(remote_url)

        tracking_url_type=urlparse(mlflow.get_tracking_uri()).scheme
        sig=infer_signature(xtrain,pred)
        if tracking_url_type!='file':
            
            mlflow.sklearn.log_model(
                lr,"model",
                registered_model_name="Simplelogisticmodel"
                ,signature=sig
            )
        else:
            mlflow.sklearn.log_model(
                lr,"model",
                signature=sig
            )

pipeline()

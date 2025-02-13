import dagshub
dagshub.init(repo_owner='SKrishna-7', repo_name='complete-mlflow-lifecycle', mlflow=True)

export MLFLOW_TRACKING_URL=https://dagshub.com/SKrishna-7/complete-mlflow-lifecycle.mlflow 

export MLFLOW_TRACKING_PASSWORD=77f7b15689b1ac246d68b75b76b77f3c467f812a 
export MLFLOW_TRACKING_USERNAME=SKrishna-7 
python app.py

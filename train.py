import json
import os
import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error as mse
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('iris.csv')
os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
Y = df['target']
X = df.drop(['target'], axis=1)
param = yaml.safe_load(open('params.yaml'))
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=param['split']['test_size'],
    random_state=param['train']['random_state'])
model = RandomForestClassifier(
    n_estimators=param['train']['n_estimators'],
    random_state=param['train']['random_state']
)
model.fit(x_train, y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
mean_s_err = mse(y_test, pred)

joblib.dump(model, "models/model.pkl")
json.dump({"accuracy": acc}, open("metrics/eval.json", "w"))
json.dump({"mean_squared_error": mean_s_err}, open("metrics/mse.json", "w"))

from sklearn.datasets import load_iris
import pandas as pd

data = load_iris(as_frame=True)
df = pd.concat([data.data, pd.Series(data.target, name='target')], axis=1)
df.to_csv('iris.csv', index=False)

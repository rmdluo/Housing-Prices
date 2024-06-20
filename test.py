import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import numpy as np

test_data = pd.read_csv('data/test.csv')
ids = test_data['Id']
test_data = test_data.drop(['Id'], axis=1)

label = LabelEncoder()
na_data = pd.read_csv('labelers/numeric.csv')

for col in test_data.columns:
    if(test_data[col].dtype=='object'):
        test_data[col] = test_data[col].fillna("NA")
        label.classes_ = np.load(f'labelers/classes_{col}.npy', allow_pickle=True)
        test_data[col] = label.transform(test_data[col])
        test_data[col] = test_data[col].astype("category")
    elif(is_numeric_dtype(test_data[col])):
        test_data[col] = test_data[col].fillna(na_data[na_data['col'] == col]['val'])

model = xgb.XGBRegressor(tree_method="hist", enable_categorical=True, eval_metric=mean_absolute_error)
model.load_model('housing_price.json')

results = model.predict(test_data)
results = pd.concat((ids, pd.DataFrame(results, columns=["SalePrice"])), axis=1)
results.to_csv("submission.csv", index=False)
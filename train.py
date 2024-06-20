import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import numpy as np

train_data = pd.read_csv('data/train.csv')
train_data = train_data.drop(["Id"], axis=1)
label=LabelEncoder()

numeric_NA_vals = []

for col in train_data.columns:
    if(train_data[col].dtype=='object'):
        train_data[col] = train_data[col].fillna("NA")
        label = label.fit(pd.concat((train_data[col], pd.Series(["NA"]))))
        np.save(f'labelers/classes_{col}.npy', label.classes_)
        # print(label.classes_)
        train_data[col] = label.transform(train_data[col])
        train_data[col] = train_data[col].astype("category")
    elif(is_numeric_dtype(train_data[col])):
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        numeric_NA_vals.append(pd.DataFrame([[str(col), train_data[col].mean()]], columns=['col', 'val']))
numeric_NA_vals = pd.concat(numeric_NA_vals, ignore_index=True)
numeric_NA_vals.to_csv("labelers/numeric.csv")
train_features = train_data.drop(['SalePrice'], axis=1)
train_labels = train_data["SalePrice"]

model = xgb.XGBRegressor(tree_method="hist", enable_categorical=True, eval_metric=mean_absolute_error)
model.fit(train_features, train_labels)
model.save_model("housing_price.json")

# Housing-Prices
Housing price regression using kaggle's [house prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Installation
```
conda env create -f environment.yml
```

## Data setup
With 7z:
```
cd data
7z x house-prices-advanced-regression-techniques.zip
```

With unzip:
```
cd data
unzip house-prices-advanced-regression-techniques.zip
```

## Train
```
python train.py
```

## Test (create kaggle submission)
```
python test.py
```

## Look at the data
Run the cells in data_exploration.ipynb!
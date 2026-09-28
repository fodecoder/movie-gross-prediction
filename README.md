# Movie Gross Prediction from Conventional and Social Media Features

Regression analysis predicting a movie's box-office gross from conventional features (budget, screens, genre, sequel) and social-media features (views, likes, dislikes, comments, sentiment).

Final project for the *Data Spaces* course, Politecnico di Torino, A.Y. 2019/2020.

## Contents

- Data cleaning and handling of missing values
- Binary encoding of the categorical `Genre` feature, min-max normalization
- Correlation analysis (Pearson)
- Regression on *Gross*: linear regression and random forest, with and without PCA
- Classification of *Net income* (gross − budget) above/below the median: SVM and logistic regression

## Dataset

[CSM (Conventional and Social Media Movies) Dataset 2014 and 2015](https://archive.ics.uci.edu/ml/datasets/CSM+%28Conventional+and+Social+Media+Movies%29+Dataset+2014+and+2015) from the UCI Machine Learning Repository (231 movies, 14 attributes). Download it and save it as `dataset.csv` next to the notebook.

## Run

```bash
pip install pandas numpy seaborn scikit-learn matplotlib jupyter
jupyter notebook movie_gross_prediction.ipynb
```

## Known limitations

Written as a student project; I would do some things differently today:

- The scaler and PCA are fitted on the whole dataset before the train/test split, so test statistics leak into training. Fitting them inside a `Pipeline` on the training fold only is the correct approach.
- With ~220 samples a single 70/30 split gives noisy scores; k-fold cross-validation would be more reliable.
- The reported "precision" of the regressors is actually the R² score.
- Binary encoding of a nominal feature adds an artificial ordering between genres; one-hot encoding is safer with this few categories.

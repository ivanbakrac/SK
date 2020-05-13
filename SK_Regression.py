import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


boston = load_boston()
boston_df = boston.data
boston_df = pd.DataFrame(boston_df, columns=boston.feature_names)
target = pd.DataFrame(boston.target, columns=["MEDV"])

X = boston_df[
    [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]
]
Y = target["MEDV"]

lm = LinearRegression()
lm.fit(X, Y)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=["Coefficient"])

coeff_df = abs(coeff_df)
coeff_df = coeff_df.sort_values(by=["Coefficient"])

print(coeff_df)
print(coeff_df[-1:])




    


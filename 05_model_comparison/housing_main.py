#housing    <회귀형>
'''
LR MAE:  0.5332001304956565
LR RMSE:  0.7455813830127763
LR R2_S:  0.575787706032451
KNN MAE:  0.4461535271317829
KNN RMSE:  0.6575877238850522
KNN R2_S:  0.6700101862970989
SVR MAE:  0.3985982830145758
SVR RMSE:  0.5974969813107398
SVR R2_S:  0.7275639524733043
RF MAE:  0.32785807386143423
RF RMSE:  0.5075567573129056
RF R2_S:  0.8034095719949859
'''

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X,y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

models = {
    "LR": make_pipeline(StandardScaler(), LinearRegression()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor()),
    "SVR": make_pipeline(StandardScaler(), SVR()),
    "RF": make_pipeline(StandardScaler(), RandomForestRegressor())
}

for model_name, model_pipeline in models.items():
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_s = r2_score(y_test, y_pred)
    print(model_name + " MAE: ", mae)
    print(model_name + " RMSE: ", rmse)
    print(model_name + " R2_S: ", r2_s)



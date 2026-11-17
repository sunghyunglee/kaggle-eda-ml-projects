#diabetes
'''
LR MAE:  42.79409467959994
LR RMSE:  53.85344583676593
LR R2_S:  0.4526027629719195
KNN MAE:  42.777528089887646
KNN RMSE:  55.203712625515045
KNN R2_S:  0.42480887066066253
SVR MAE:  56.0294733040062
SVR RMSE:  65.82354046498651
SVR R2_S:  0.18221699094239507
RF MAE:  44.2770786516854
RF RMSE:  54.69885071610665
RF R2_S:  0.435281507272678
'''

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# data= load_diabetes()
# print(data.target_names)    #회귀인걸 알아냄

X,y = load_diabetes(return_X_y=True)

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


import warnings

from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from scipy.stats import skew
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

house_df_org = pd.read_csv('C:\\Users\\Admin\\Desktop\\데이터 분석\\대표 datasets\\house_price\\houseprice.csv')
house_df = house_df_org.copy()
house_df.head()
house_df.info()

house_df.shape
house_df.dtypes
house_df.dtypes.value_counts()

isnull_series = house_df.isnull().sum()
print(isnull_series[isnull_series > 0].sort_values(ascending = False))

plt.title('Original Sale Price Histogram')
sns.distplot(house_df['SalePrice']) # 한쪽으로 치우친 그래프
plt.show()

plt.title('Log Transformed Sale Price Histogram')
log_SalePrice = np.log1p(house_df['SalePrice'])
sns.distplot(log_SalePrice)
plt.show()

original_SalePrice = house_df['SalePrice'] # 일단 원본 살려놓음
house_df['SalePrice'] = np.log1p(house_df['SalePrice'])
# 결측치 너무 많은 것 제거()
house_df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1, inplace = True)
house_df.fillna(house_df.mean(), inplace = True) # 수치형 데이터만 결측제거함

null_column_count = house_df.isnull().sum()[house_df.isnull().sum() > 0]
null_column_count
print('## Null 피처의 Type: \n', house_df.dtypes[null_column_count.index])
null_column_count.index # 컬럼 이름

print('get_dummies() 수행 전 데이터 shape: ', house_df.shape) # 75
house_df_ohe = pd.get_dummies(house_df)
print('get_dummies() 수행 후 데이터 shape: ', house_df_ohe.shape) # 271
house_df_ohe.info()

null_column_count = house_df_ohe.isnull().sum()[house_df_ohe.isnull().sum() > 0]
print('## Null 피처의 Type: \n', house_df.dtypes[null_column_count.index])

def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(model.__class__.__name__, '로그 변환된 RMSE: ', np.round(rmse, 3))
    return mse

def get_rmses(models):
    rmses = []
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses

y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice', axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size = 0.2, random_state = 156)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

def get_top_bottom_coef(model, n = 10):
    coef = pd.Series(model.coef_, index = X_features.columns)
    coef_high = coef.sort_values(ascending = False).head(n)
    coef_low = coef.sort_values(ascending = False).tail(n)
    return coef_high, coef_low

def visualize_coefficient(models):
    fig, axs = plt.subplots(figsize = (24, 10), nrows = 1, ncols = 3)
    fig.tight_layout()

    for i, model in enumerate(models):
        coef_high, coef_low = get_top_bottom_coef(model)
        coef_concat = pd.concat([coef_high, coef_low])

        axs[i].set_title(model.__class__.__name__ + ' Coefficients', size = 25)
        axs[i].tick_params(axis = 'y', direction = 'in', pad = -120)

        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x = coef_concat.values, y = coef_concat.index, ax = axs[i])

visualize_coefficient(models)
plt.show()

def get_avg_rmse_cv(models):
    for model in models:
        rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target, scoring = "neg_mean_squared_error", cv = 5))
        rmse_avg = np.mean(rmse_list)
        print('\n{0} CV RMSE 값 리스트: {1}'.format(model.__class__.__name__, np.round(rmse_list, 3)))
        print('{0} CV 평균 RMSE 값: {1}'.format(model.__class__.__name__, np.round(rmse_avg, 3)))

get_avg_rmse_cv(models)

def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid = params, scoring = 'neg_mean_squared_error', cv = 5)
    grid_model.fit(X_features, y_target)
    rmse = np.sqrt(-1 * grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha: {2}'.format(model.__class__.__name__, np.round(rmse, 4), grid_model.best_params_))

ridge_params = {'alpha': [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}
lasso_params = {'alpha': [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]}

print_best_params(ridge_reg, ridge_params)
print_best_params(lasso_reg, lasso_params)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha = 12)
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

visualize_coefficient(models)
plt.show()

## 왜도를 보자(치우침 정도)
feature_index = house_df.dtypes[house_df.dtypes != "object"].index
feature_index # 컬럼 이름
skew_features = house_df[feature_index].apply(lambda x: skew(x))
skew_features

skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending = False))

house_df[skew_features_top.index] = np.log1p(house_df[skew_features_top.index])

house_df_ohe = pd.get_dummies(house_df)
y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop(['SalePrice'], axis = 1, inplace = False)
X_train, X_test, y_train, y_target = train_test_split(X_features, y_target, test_size = 0.2, random_state = 156)
X_train.shape, X_test.shape, y_train.shape, y_target.shape # 270

ridge_params = {'alpha': [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}
lasso_params = {'alpha': [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]}

print_best_params(ridge_reg, ridge_params) # X_features 실행하고 나니 갑자기 실행되네...?
print_best_params(lasso_reg, lasso_params) # 차례대로 실행해야 되나 보네....

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha = 10)
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

visualize_coefficient(models)
plt.show()

# 다른 독립변수와 산점도, 이상치 판단
plt.scatter(house_df_org['GrLivArea'], house_df_org['SalePrice'])
plt.xlabel('GrLivArea', fontsize = 15)
plt.ylabel('SalePrice', fontsize = 15)
plt.show()

np.max(house_df_ohe['GrLivArea']) # 8.63817111796914
np.log1p(4000) # 8.294299608857235
np.max(house_df_ohe['SalePrice']) # 13.534474352733596
np.log1p(500000) # 13.122365377402328

cond1 = house_df_ohe['GrLivArea'] > np.log1p(4000)
cond2 = house_df_ohe['SalePrice'] < np.log1p(500000)
outlier_index = house_df_ohe[cond1 & cond2].index

print('이상치 레코드 index: ', outlier_index.values)
print('이상치 삭제 전 house_df_ohe shape: ', house_df_ohe.shape)

house_df_ohe.drop(outlier_index, axis = 0, inplace = True)
print('이상치 삭제 후 house_df_ohe shape: ', house_df_ohe.shape)

y_target = house_df_ohe['SalePrice']
X_feature = house_df_ohe.drop('SalePrice', axis = 1, inplace = False)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size = 0.2, random_state = 156)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

print_best_params(ridge_reg, ridge_params)
print_best_params(lasso_reg, lasso_params)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha = 8)
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

visualize_coefficient(models)
plt.show()

### 모델학습 ###
xgb_params = {'n_estimators': [1000]}
xgb_reg = XGBRegressor(n_estimators = 1000, learning_rate = 0.5, colsample_bytree = 0.5, subsample = 0.8)
print_best_params(xgb_reg, xgb_params)

lgbm_params = {'n_estimators': [1000]}
lgbm_reg = LGBMRegressor(n_estimators = 1000, learning_rate = 0.05, num_leaves = 4, subsample = 0.6, colsample_bytree = 0.4, 
                        reg_lambda = 10, n_jobs = -1)
print_best_params(lgbm_reg, lgbm_params)

def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test, pred_value)
        rmse = np.sqrt(mse)
        print('{0} 모델의 RMSE: {1}'.format(key, rmse))

ridge_reg = Ridge(alpha = 8)
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha = 0.001)
lasso_reg.fit(X_train, y_train)

ridge_pred = ridge_reg.predict(X_test)
lasso_pred = lasso_reg.predict(X_test)

pred = 0.4 * ridge_pred + 0.6 * lasso_pred
preds = {'최종 혼합': pred,
        'Ridge': ridge_pred,
        'Lasso': lasso_pred}
get_rmse_pred(preds)

xgb_reg.fit(X_train, y_train)
lgbm_reg.fit(X_train, y_train)

xgb_pred = ridge_reg.predict(X_test)
lgbm_pred = lasso_reg.predict(X_test)

pred = 0.5 * xgb_pred + 0.5 * lgbm_pred
preds = {'최종 혼합': pred,
        'XGB': xgb_pred,
        'LGBM': lgbm_pred}

get_rmse_pred(preds)

def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    kn = KFold(n_splits = n_folds, shuffle = True, random_state = 0)

    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(model.__class__.__name__, 'model 시작')

    for folder_counter, (train_index, valid_index) in enumerate(kn.split(X_train_n)):
        # 입력된 학습 데이터에서 모델이 학습/예측할 폴드 데이터 세트 추출
        print('\t 폴드 세트: ', folder_counter, ' 시작')
        X_tr = X_train_n[train_index]
        y_tr = y_train_n[train_index]
        X_te = X_train_n[valid_index]

        # 폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행
        model.fit(X_tr, y_tr)
        # 폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1, 1)
        # 입력된 원본 테스트 데이터를 폴드 세트 내 학습된 모델에서 예측 후 데이터 저장
        test_pred[:, folder_counter] = model.predict(X_test_n)
    
    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성
    test_pred_mean = np.mean(test_pred, axis = 1).reshape(-1, 1)

    # train_fold_pred는 최종 메타 모델이 사용하는 학습데이터, test_pred_mean은 테스트 데이터
    return train_fold_pred, test_pred_mean 

X_train_n = X_train.values
X_test_n = X_test.values
y_train_n = y_train.values

ridge_train, ridge_test = get_stacking_base_datasets(ridge_reg, X_train_n, y_train_n, X_test_n, 5)
lasso_train, lasso_test = get_stacking_base_datasets(lasso_reg, X_train_n, y_train_n, X_test_n, 5)
xgb_train, xgb_test = get_stacking_base_datasets(xgb_reg, X_train_n, y_train_n, X_test_n, 5)
lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm_reg, X_train_n, y_train_n, X_test_n, 5)

Stack_final_X_train = np.concatenate((ridge_train, lasso_train, xgb_train, lgbm_train), axis = 1)
Stack_final_X_test = np.concatenate((ridge_test, lasso_test, xgb_test, lgbm_test), axis = 1)

meta_model_lasso = Lasso(alpha = 0.0005)
meta_model_lasso.fit(Stack_final_X_train, y_train)

final = meta_model_lasso.predict(Stack_final_X_test)
mse = mean_squared_error(y_test, final)
rmse = np.sqrt(mse)
print('스태킹 회귀 모델의 최종 RMSE 값: ', round(rmse, 4))
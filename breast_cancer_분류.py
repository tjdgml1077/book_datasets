import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
import xgboost as xgb # 파이썬 래퍼
from xgboost import plot_importance
from xgboost import XGBClassifier # 사이킷런 래퍼
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
X_features = dataset.data
y_label = dataset.target

cancer_df = pd.DataFrame(data = X_features, columns = dataset.feature_names)
cancer_df['target'] = y_label
cancer_df.head()

print(dataset.target_names)
print(cancer_df['target'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size = 0.2, random_state = 156)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

### 파이썬 래퍼 XGBoost ###
dtrain = xgb.DMatrix(data = X_train, label = y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)

params = {'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping': 100}
num_rounds = 400

wlist = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params = params, dtrain = dtrain, num_boost_round = num_rounds, \
                        early_stopping_rounds = 100, evals = wlist) # fit대신 train사용함

pred_probs = xgb_model.predict(dtest)
print(np.round(pred_probs[:10], 3))

preds = [1 if x > 0.5 else 0 for x in pred_probs]
print(preds[:10])

def get_clf_eval(y_test, pred = None, pred_proba = None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print('오차행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, f1 스코어: {3:.4f}, ROC-AUC score: {4:.4f}'\
        .format(accuracy, precision, recall, f1, roc_auc))

get_clf_eval(y_test, preds, pred_probs)

fig, ax = plt.subplots(figsize = (10, 12))
plot_importance(xgb_model, ax = ax)
plt.show()

### 사이킷런 래퍼 xgboost ###
xgb_wrapper = XGBClassifier(n_estimators = 400, 
                            learning_rate = 0.1, 
                            max_depth = 3)
xgb_wrapper.fit(X_train, y_train)
w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

get_clf_eval(y_test, w_preds, w_pred_proba)

cancer_df.shape # (569, 31) 데이터 세트의 크기가 너무 작아 부득이하게 테스트 데이터를 평가용으로 사용함

evals = [(X_test, y_test)]
xgb_wrapper.fit(X_train, y_train, early_stopping_rounds = 100, eval_metric = 'logloss', eval_set = evals, verbose = True)
                 # 평가지표가 향상될 수 있는 반복횟수 지정(조기중단)
                 # 성능평가를 수행할 데이터세트 eval_set(완전히 알려지지 않은 새로운 데이터세트를 사용하는 것이 바람직)
                 # eval_metric의 logloss(Negative log-likelihood)
                 # verbose = True는 함수 수행 시 발생하는 상세한 정보들을 출력으로 자세히 내보낸다는 뜻
ws100_preds =xgb_wrapper.predict(X_test)
ws100_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(figsize = (10, 12))
plot_importance(xgb_wrapper, ax = ax)
plt.show() # 결과는 같음

### LightGBM ###
lgbm_wrapper = LGBMClassifier(n_estimators = 400)
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds = 100, eval_metric = 'logloss', eval_set = evals, verbose = True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)

### 기본 스태킹 ###
knn_clf = KNeighborsClassifier(n_neighbors = 4)
rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 0)
dt_clf = DecisionTreeClassifier()
ada_clf = AdaBoostClassifier(n_estimators = 100)

lr_final = LogisticRegression(C = 10)

knn_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
ada_clf.fit(X_train, y_train)

knn_pred = knn_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
dt_pred = dt_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)

print('KNN 정확도: {0:.4f}'.format(accuracy_score(y_test, knn_pred)))
print('Randomforest 정확도: {0:.4f}'.format(accuracy_score(y_test, rf_pred)))
print('DecisionTree 정확도: {0:.4f}'.format(accuracy_score(y_test, dt_pred)))
print('Adaboost 정확도: {0:.4f}'.format(accuracy_score(y_test, ada_pred)))

pred = np.array([knn_pred, rf_pred, dt_pred, ada_pred])
print(pred.shape)
pred = np.transpose(pred)
print(pred.shape)

lr_final.fit(pred, y_test) # 좌우 바뀌는 구나...
final = lr_final.predict(pred)
print('최종 메타 모델 예측 정확도: {0:.4f}'.format(accuracy_score(y_test, final)))

### cv 세트 기반 스태킹 ###
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    kf = KFold(n_splits = n_folds, shuffle = True, random_state = 0)

    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(model.__class__.__name__, 'model 시작')

    for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        print('\t 폴드 세트: ', folder_counter, ' 시작')
        X_tr = X_train_n[train_index]
        y_tr = y_train_n[train_index]
        X_te = X_train_n[valid_index]

        model.fit(X_tr, y_tr)

        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1, 1)
        # valid_index 번째 행 값은 X_te의 예측값(1행의 값이므로 reshape)
        test_pred[:, folder_counter] = model.predict(X_test_n)
        # folder_counter 번째 열의 값은 X_test_n의 예측값

    test_pred_mean = np.mean(test_pred, axis = 1).reshape(-1, 1)

    return train_fold_pred, test_pred_mean

knn_train, knn_test = get_stacking_base_datasets(knn_clf, X_train, y_train, X_test, 7)
rf_train, rf_test = get_stacking_base_datasets(rf_clf, X_train, y_train, X_test, 7)
dt_train, dt_test = get_stacking_base_datasets(dt_clf, X_train, y_train, X_test, 7)
ada_train, ada_test = get_stacking_base_datasets(ada_clf, X_train, y_train, X_test, 7)

Stack_final_X_train = np.concatenate((knn_train, rf_train, dt_train, ada_train), axis = 1)
Stack_final_X_test = np.concatenate((knn_test, rf_test, dt_test, ada_test), axis = 1)
print('원본 학습 feature 데이터 shape: ', X_train.shape, '원본 테스트 feature 데이터 shape: ', X_test.shape)
print('스태킹 학습 feature 데이터 shape: ', Stack_final_X_train.shape, '스태킹 테스트 feature 데이터 shape: ', Stack_final_X_test.shape)

lr_final.fit(Stack_final_X_train, y_train)
stack_final = lr_final.predict(Stack_final_X_test)

print('최종 메타 모델의 예측 정확도: {0:.4f}'.format(accuracy_score(y_test, stack_final)))



##### Scaling 했을 때 #####
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, cancer.target, test_size = 0.3, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_preds = lr_clf.predict(X_test)

print('accuaracy: {0:.4f}'.format(accuracy_score(y_test, lr_preds)))
print('roc_auc: {0:.4f}'.format(roc_auc_score(y_test, lr_preds)))

from sklearn.model_selection import GridSearchCV

params = {'penalty': ['l2', 'l1'],
        'C': [0.01, 0.1, 1, 5, 10]}
grid_clf = GridSearchCV(lr_clf, param_grid = params, scoring = 'accuracy', cv = 3)
grid_clf.fit(data_scaled, cancer.target)
print('최적 하이퍼파라미터: {0}, 최적 평균 정확도: {1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))


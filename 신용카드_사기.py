# 신용카드 고객과 다른 것
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

card_df = pd.read_csv("C:\\Users\\Admin\\Desktop\\데이터 분석\\대표 datasets\\신용카드_사기\\creditcard.csv")
card_df.head()

def get_preprocessed_df(df = None):
    df_copy = df.copy()
    df_copy.drop('Time', axis = 1, inplace = True)
    return df_copy

def get_train_test_dataset(df = None):
    df_copy = get_preprocessed_df(df)
    X_features = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size = 0.3, random_state = 0, stratify = y_target)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

y_train.value_counts()
print(y_train.value_counts() / y_train.shape[0] * 100)
print(y_test.value_counts() / y_test.shape[0] * 100)

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
lr_pred_proba = lr_clf.predict_proba(X_test)[:, 1]

def get_clf_eval(y_test, pred = None, pred_proba = None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, f1 스코어: {3:.4f}, ROC-AUC score: {4:.4f}' \
        .format(accuracy, precision, recall, f1, roc_auc))

get_clf_eval(y_test, lr_pred, lr_pred_proba)

def get_model_train_eval(model, ftr_train = None, ftr_test = None, tgt_train = None, tgt_test = None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:, 1]
    get_clf_eval(tgt_test, pred, pred_proba)

lgbm_clf = LGBMClassifier(n_estimators = 1000,
                        num_leaves = 64,
                        n_jobs = -1,
                        boost_from_average = False)
get_model_train_eval(lgbm_clf, X_train, X_test, y_train, y_test)




plt.figure(figsize = (8, 4))
plt.xticks(range(0, 30000, 1000), rotation = 60)
sns.distplot(card_df['Amount']) # 한쪽으로 치우친 비대칭
plt.show()

# 위에는 표준화 없이 한거, 이거는 표준화!!
# StandardScaler() 적용
def get_preprocessed_df(df = None): 
    df_copy = df.copy()
    scaler = StandardScaler()
    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1, 1))
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time', 'Amount'], axis = 1, inplace = True)
    return df_copy

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('로지스틱 예측 성능')
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train, X_test, y_train, y_test)

print('LGBM 예측 성능')
lgbm_clf = LGBMClassifier(n_estimators = 1000,
                        num_leaves = 64,
                        n_jobs = -1, 
                        boost_from_average = False)
get_model_train_eval(lgbm_clf, ftr_train = X_train,  ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)

### log1p 적용
def get_preprocessed_df(df = None): 
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time', 'Amount'], axis = 1, inplace = True)
    return df_copy

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('로지스틱 예측 성능')
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train, X_test, y_train, y_test)

print('LGBM 예측 성능')
lgbm_clf = LGBMClassifier(n_estimators = 1000,
                        num_leaves = 64,
                        n_jobs = -1, 
                        boost_from_average = False)
get_model_train_eval(lgbm_clf, ftr_train = X_train,  ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)





plt.figure(figsize = (9, 9))
corr = card_df.corr()
sns.heatmap(corr, cmap = 'RdBu')
plt.show()

### 이상치 ###
card_df['Class'].value_counts()
def get_outlier(df = None, column = None, weight = 1.5):
    fraud = df[df['Class'] == 1][column] # 1일 때 사기당함
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight

    # 이상치의 인덱스
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    return outlier_index

outlier_index = get_outlier(card_df, column = 'V14', weight = 1.5)
print('이상치 데이터 인덱스: ', outlier_index)

### log1p 다시 적용
def get_preprocessed_df(df = None): 
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time', 'Amount'], axis = 1, inplace = True)

    outlier_index = get_outlier(df = card_df, column = 'V14', weight = 1.5)
    df_copy.drop(outlier_index, axis = 0, inplace = True)

    return df_copy

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)

print('로지스틱 예측 성능')
lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train, X_test, y_train, y_test)

print('LGBM 예측 성능')
lgbm_clf = LGBMClassifier(n_estimators = 1000,
                        num_leaves = 64,
                        n_jobs = -1, 
                        boost_from_average = False)
get_model_train_eval(lgbm_clf, ftr_train = X_train,  ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)

### SMOTE ###
smote = SMOTE(random_state = 0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train) # 주피터는 sample을 씀
print(X_train.shape, y_train.shape)
print(X_train_over.shape, y_train_over.shape)

print(pd.Series(y_train_over).value_counts())
y_train.value_counts() # 0에 몰린 과대적합

lr_clf = LogisticRegression()
get_model_train_eval(lr_clf, X_train_over, X_test, y_train_over, y_test)

def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    plt.figure(figsize = (8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle = '--', label = 'precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label = 'recall')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlabel('threshold value')
    plt.ylabel('precision and recall value')
    plt.legend(); plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, lr_clf.predict_proba(X_test)[:, 1])

lgbm_clf = LGBMClassifier(n_estimators = 1000, num_leaves = 64, n_jobs = -1, boost_from_average = False)
get_model_train_eval(lgbm_clf, ftr_train = X_train_over,  ftr_test = X_test, tgt_train = y_train_over, tgt_test = y_test)


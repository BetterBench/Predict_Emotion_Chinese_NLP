import pandas as pd
import re
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score


df = pd.read_csv('data/clear_data.csv')
train_text = df["Text"]
train_label = df["label"]
col = df.columns[-9:]
emo_note = df[col].values

# 将语料转化为词袋向量，根据词袋向量统计TF-IDF
vectorizer = CountVectorizer(max_features=1000)
tf_idf_transformer = TfidfTransformer()
tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(train_text))
X_weight = tf_idf.toarray()  # 训练集TF-IDF权重矩阵
# 将9个情感标注的label作为模型的输入特征
train_data = np.hstack((X_weight, emo_note))
# 划分10折
NFOLDS = 10
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=1)
kf = kfold.split(train_data, train_label)
valid_best = 0
model = LGBMClassifier(n_estimators=500)
for i, (train_fold, validate) in enumerate(kf):

    X_train, X_validate, label_train, label_validate = (
        train_data[train_fold],
        train_data[validate],
        train_label[train_fold],
        train_label[validate],
    )
    model.fit(X_train, label_train)
    y_prd= model.predict(X_validate,)
    score = accuracy_score(label_validate, y_prd)
    print('score ={}'.format(score))



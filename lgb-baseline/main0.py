# %%

import warnings

warnings.simplefilter('ignore')

import os
import gc
import re
import glob

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, auc

from urllib.parse import quote, unquote, urlparse

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]

# %%


# %%

import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化


set_seed(2024)

# %%

# train

train_files = glob.glob('../data/train/*.csv')

df_train = pd.DataFrame() # 创建框架并赋值给df_train

for filepath in tqdm(train_files):
    df = pd.read_csv(filepath)
    df_train = pd.concat([df_train, df]).reset_index(drop=True)

df_train.fillna('__NaN__', inplace=True) # 合并文件

# 强迫症发作..
df_train = df_train.rename(columns={'lable': 'label'})
df_train
print(len(df_train))

# %%

# label
# 0. 白
# 1. SQL 注入
# 2. 目录历遍
# 3. 远程代码执行
# 4. 命令执行
# 5. XSS 跨站脚本

# %%

df_test = pd.read_csv('../data/test/test.csv')
df_test.fillna('__NaN__', inplace=True)
df_test

# %%

df = pd.concat([df_train, df_test]).reset_index(drop=True) # 合并train与test并返回新数据的形状
df.shape


# %%

def get_url_query(s):
    li = re.split('[=&]', urlparse(s)[4])
    return [li[i] for i in range(len(li)) if i % 2 == 1]
# https://example.com/page?name=John&age=25&city=NewYork"['John', '25', 'NewYork']

def find_max_str_length(x):
    max_ = 0
    li = [len(i) for i in x]
    return max(li) if len(li) > 0 else 0 #返回给定字符数组中最长字符的长度


def find_str_length_std(x):
    max_ = 0
    li = [len(i) for i in x]
    return np.std(li) if len(li) > 0 else -1 # np.std: 返回标准差；若列表为空，返回-1

# 对df中的url列进行提取特征，结果储存在新的列中
df['url_unquote'] = df['url'].apply(unquote)# 解码，将一些特殊符号转化成%25形式
df['url_query'] = df['url_unquote'].apply(lambda x: get_url_query(x)) # 提取url里的查询参数
df['url_query_num'] = df['url_query'].apply(len) # 存储每个url里查询参数的个数
df['url_query_max_len'] = df['url_query'].apply(find_max_str_length) # 每个url查询参数的最大长度
df['url_query_len_std'] = df['url_query'].apply(find_str_length_std) # 查询参数的长度的变化程度(标准差)


# %%

def find_url_filetype(x):
    try:
        return re.search(r'\.[a-z]+', x).group()# 匹配文件扩展名(.后的字符
    except:
        return '__NaN__'


df['url_path'] = df['url_unquote'].apply(lambda x: urlparse(x)[2]) # 用urlparse提取其路径
df['url_filetype'] = df['url_path'].apply(lambda x: find_url_filetype(x)) # 保存文件扩展名

df['url_path_len'] = df['url_path'].apply(len) # 路径的长度
df['url_path_num'] = df['url_path'].apply(lambda x: len(re.findall('/', x))) # url里的斜杠数量

# %%
# user-agent信息提取
df['ua_short'] = df['user_agent'].apply(lambda x: x.split('/')[0]) # 斜杠分割后的第一个部分，也就是浏览器类型
df['ua_first'] = df['user_agent'].apply(lambda x: x.split(' ')[0]) # 浏览器或设备名称
# Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36


# %%

# %%time

# TF-IDF特征添加、降维、提取特征
def add_tfidf_feats(df, col, n_components=16):
    text = list(df[col].values)
    tf = TfidfVectorizer(min_df=1, # 最小词频1  # 创建一个TF-IDF对象
                         analyzer='char_wb', # 字符级别的分析
                         ngram_range=(1, 3), # 1-3的ngram特征: 允许长度1、2、3的字符序列
                         stop_words='english') # 设置停用词列表
    tf.fit(text) # 进行拟合(1、接收词汇表，2、计算TF-IDF)
    X = tf.transform(text) # 将转化为TF-IDF特征矩阵x
    svd = TruncatedSVD(n_components=n_components) # 截断奇异值分解(高阶矩阵->低阶)
    svd.fit(X)
    X_svd = svd.transform(X)
    for i in range(n_components):
        df[f'{col}_tfidf_{i}'] = X_svd[:, i]
    return df


df = add_tfidf_feats(df, 'url_unquote', n_components=16)
df = add_tfidf_feats(df, 'user_agent', n_components=16)
df = add_tfidf_feats(df, 'body', n_components=32)

# %%

# 使用 LabelEncoder 对 DataFrame 中的一些分类特征进行编码
for col in tqdm(['method', 'refer', 'url_filetype', 'ua_short', 'ua_first']):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# %%

not_use_feats = ['id', 'user_agent', 'url', 'body', 'url_unquote', 'url_query', 'url_path', 'label']# 后续特征中不会被考虑
use_features = [col for col in df.columns if col not in not_use_feats]

# %%

train = df[df['label'].notna()]
test = df[df['label'].isna()]

train.shape, test.shape

# %%

NUM_CLASSES = 6
FOLDS = 5
TARGET = 'label'

from sklearn.preprocessing import label_binarize


def run_lgb(df_train, df_test, use_features):
    target = TARGET
    oof_pred = np.zeros((len(df_train), NUM_CLASSES))
    y_pred = np.zeros((len(df_test), NUM_CLASSES))

    folds = StratifiedKFold(n_splits=FOLDS)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train[TARGET])):
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)

        params = {
            'learning_rate': 0.1,
            'metric': 'multiclass',
            'objective': 'multiclass',
            'num_classes': NUM_CLASSES,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 2,
            'n_jobs': -1,
            'seed': 2022,
            'max_depth': 10,
            'num_leaves': 100,
            'lambda_l1': 0.5,
            'lambda_l2': 0.8,
            'verbose': -1
        }

        model = lgb.train(params,
                          train_set,
                          num_boost_round=500,
                          #early_stopping_rounds=100,
                          valid_sets=[train_set, val_set],
                          callbacks=callbacks)
                          #verbose_eval=100)
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(df_test[use_features]) / folds.n_splits

        print("Features importance...")
        gain = model.feature_importance('gain')
        feat_imp = pd.DataFrame({'feature': model.feature_name(),
                                 'split': model.feature_importance('split'),
                                 'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        print('Top 50 features:\n', feat_imp.head(50))

        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()

    return y_pred, oof_pred


y_pred, oof_pred = run_lgb(train, test, use_features)

# %%

print(accuracy_score(np.argmax(oof_pred, axis=1), df_train['label']))

# %%

sub = pd.read_csv('../data/submit_example.csv')
sub['predict'] = np.argmax(y_pred, axis=1)
sub

# %%

sub['predict'].value_counts()

# %%

sub.to_csv('main0.csv', index=False)

# %%



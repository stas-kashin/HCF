import numpy as np
import pandas as pd
import gc
import os
import sys
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV, cross_val_score   #Perforing grid search
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


INPUT_FILES_FOLDER, OUTPUT_FILES_FOLDER, OUTPUT_FILES_PREFIX = "data/input/", \
                                                               "data/output/"+time.strftime("%Y%m%d/", \
                                                                time.gmtime()), time.strftime("%Y%m%d_%H%M%S", time.gmtime())


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx],
                             label=train_df['TARGET'].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx],
                             label=train_df['TARGET'].iloc[valid_idx],
                             free_raw_data=False, silent=True)

        # LightGBM parameters found by Bayesian optimization
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 4,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60,  # 39.3259775,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
        }

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        oof_preds[valid_idx] = clf.predict(dvalid.data)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(dvalid.label, oof_preds[valid_idx])))
        del clf, dtrain, dvalid
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        sub_df = test_df[['SK_ID_CURR']].copy()
        sub_df['TARGET'] = sub_preds
        sub_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    # return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(24, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout
    plt.savefig(OUTPUT_FILES_FOLDER+OUTPUT_FILES_PREFIX+'lgbm_importances.png')


def main(debug=False):
    num_rows = 10000 if debug else None

    with open(INPUT_FILES_FOLDER+"train_test_rev_dataset.csv", 'r') as f:
        df = pd.read_csv(f, nrows=num_rows)

    with timer("Run LightGBM with kfold"):
        print(df.shape)
        # df.drop(features_with_no_imp_at_least_twice, axis=1, inplace=True)
        gc.collect()
        # print(df.shape)
        # feat_importance = kfold_lightgbm(df, num_folds=5, stratified=False, debug=debug)

        # TEST \/
        # kfold_lightgbm(df, num_folds=5, stratified=False, debug=debug)

        train_df = df[df['TARGET'].notnull()]
        test_df = df[df['TARGET'].isnull()]
        print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
        del df
        gc.collect()

        target = 'TARGET'
        IDcol = 'SK_ID_CURR'

        predictors = [x for x in train_df.columns if x not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

        # Baseline
        # gbm0 = lgb.LGBMClassifier(random_state=67, objective='binary')
        # modelfit(gbm0, train_df, predictors, target)

        param_test1 = {'n_estimators': [70, 90, 120],  'num_leaves':[5, 10, 15, 20], 'learning_rate':[0.095, 0.097, 0.1]}
        gsearch1 = GridSearchCV(
            estimator=lgb.LGBMClassifier(n_estimators=120, num_leaves=5, learning_rate=0.095, max_depth=8, objective='binary', subsample=0.8, random_state=67),
            param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        gsearch1.fit(train_df[predictors], train_df[target])
        print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
        # TEST /\



# TEST \/

def modelfit(alg, dtrain, predictors, target, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=cv_folds, scoring='roc_auc')

    # Print model report:
    print("\nModel Report\n")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("\nAUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g\n" % \
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance:
    if printFeatureImportance:
        plt.figure(figsize=(48, 40))
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.tight_layout
        plt.savefig(OUTPUT_FILES_FOLDER + OUTPUT_FILES_PREFIX + 'lgbm_importances.png')


# TEST /\

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FILES_FOLDER):
        os.makedirs(OUTPUT_FILES_FOLDER)

    submission_file_name = OUTPUT_FILES_FOLDER+OUTPUT_FILES_PREFIX+"prediction_rev.csv"
    log_file_name = OUTPUT_FILES_FOLDER+OUTPUT_FILES_PREFIX+"output.log"

    with open(log_file_name, 'w') as log, timer("Full model run"):
        sys.stdout = log
        main()
        # log.close()

# imports
import os
import numpy as np
import pandas as pd
import math
import sys
import time

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from statistics import mean
from statistics import stdev

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score

from sklearn.linear_model import LassoLarsCV, SGDRegressor, LogisticRegression, ElasticNetCV, LinearRegression, RidgeCV, BayesianRidge

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RepeatedKFold, KFold

from .ag_datasets import generate_sparse_dataset
from .imputers import CustomIterativeImputer, ClipTransformer

import statsmodels.stats.api as sms


def calc_error(y_pred,
               y_true,
               adj_r2=False,
               crit_thresh=None,
               train_time=None,
               pred_time=None,
               missing_mask=None
):
    '''
    Inputs
        X_pred: ndarray
                missing values filled in by prediction function
        y_true: ndarray, shape == X_pred.shape
                contains all ground truth imputed_values
        missing_mask: boolean ndarray, shape == X_pred.shape
                mask of all values that were originally missing from X and subsequently
                imputed. We only want to look at these because otherwise we will
                overestimate model performance depending on the % of points dropped.

    Returns: Dictionary of various error functions applied to all imputed values
    '''


    # if y_true is a pandas dataframe, convert to numpy Array
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    # If a mask of missing values was passed, only calculate error for those values
    if missing_mask is not None:
        y_pred = y_pred[missing_mask]
        y_true = y_true[missing_mask]

    error_dict = {
        'Mean Absolute Error' : mean_absolute_error(y_pred, y_true),
        'Root Mean Squared Error' : mean_squared_error(y_pred, y_true, squared=False),
        'R2 Score' : r2_score(y_pred, y_true),

    }

    if train_time is not None:
        error_dict['Train Runtime'] = train_time

    if pred_time is not None:
        error_dict['Prediction Runtime'] = pred_time

    # Identify the number of predictions that were off by more than specified thresholds
    if crit_thresh is not None:
        abs_errors = np.absolute(y_pred - y_true)

        for thresh in crit_thresh:
            num_crit_fails = (abs_errors > thresh).sum()
            perc_crit_fails = (num_crit_fails / np.size(y_pred)) * 100
            error_dict['% Misses > ' + str(thresh)] = perc_crit_fails

    if adj_r2:
        # Calculate adjusted R2 score
        r2 = r2_score(y_pred, y_true)
        n_samples = missing_mask.shape[0]
        n_missing = missing_mask.any(axis=0).sum()
        n_predictors = missing_mask.shape[1] - n_missing
        adj_r2 =  1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1))
        error_dict['Adjusted R2 Score'] = adj_r2

    return error_dict


def imputer_gridsearch(estimator, X, y, params, n_gridsearch_cv):

    param_names = []
    param_vals = []

    for param_name in params:
        param_names.append(param_name)
        param_vals.append(np.array(params[param_name]))

    grid = np.meshgrid(*(x for x in param_vals))

    if n_gridsearch_cv > X.shape[0]:
        n_gridsearch_cv = X.shape[0]

    kf_ = KFold(n_splits=n_gridsearch_cv)

    mean_scores = np.zeros(shape=grid[0].shape)

    best_err = 1e10
    best_params = {}

    for idx, value in np.ndenumerate(grid[0]):

        param_dict = {}

        counter = 0
        for param_name in param_names:
            param_dict[param_name] = grid[counter][idx]
            counter +=1

        estimator.set_params(**param_dict)
        scores = []

        for train_index_, test_index_ in kf_.split(X):

            X_train_, X_test_ = X.iloc[train_index_], X.iloc[test_index_]
            y_train_, y_test_ = y.iloc[train_index_], y.iloc[test_index_]
            n_test_ = X_test_.shape[0]

            # Concatenate X_test and y_train because imputers use existing datapoints during the imputation
            X_test_ = pd.concat((X_test_, y_train_), axis=0)


            y_pred_ = estimator.fit_transform(X_test_)
            y_pred_ = y_pred_[0:n_test_, :] # trim off "helper" datapoints


            scores.append(mean_squared_error(y_test_, y_pred_))

        param_err = mean(scores)

        mean_scores[idx] = param_err

        if param_err < best_err:
            best_err = param_err
            best_params = param_dict

    estimator = estimator.set_params(**best_params)

    #print(mean_scores)

    return estimator, best_params, mean_scores


def simulate_estimations(
    X,
    y,
    outtype,
    estimator,
    modelname,
    params,
    n_sim,
    n_sim_cv,
    n_gridsearch_cv,
    gs_scoring,
    output_ndigits=2,
    random_state=0,
    verbose=0
):

    # Initialize list of error outputs. Will be concatenated into error dataframe as output.
    err_list = []

    for sim in range(0, n_sim):

        kf = KFold(n_splits=n_sim_cv, shuffle=True)

        # y_tests = []
        # y_preds = []

        fold_count = 0
        for train_index, test_index in kf.split(X):


            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_train = np.squeeze(y_train).to_numpy()
            y_test = np.squeeze(y_test).to_numpy()

            if len(params) > 0:
                est_opt = GridSearchCV(estimator=estimator,
                                         param_grid=params,
                                         cv=min(n_gridsearch_cv, X_train.shape[0]),
                                         n_jobs=-1,
                                         scoring=gs_scoring,
                                         verbose=0)
                est_opt = est_opt.fit(X_train, y_train)
                best_params = est_opt.best_params_

                if verbose >= 2:

                    print('{modelname} (S{sim}/F{fold_count}) params:{params}'.format(modelname=modelname,
                                                                                      sim=sim,
                                                                                      fold_count=fold_count,
                                                                                      params=best_params))
                if verbose >= 3:
                    print('GS errs: ', est_opt.cv_results_['mean_test_score'])

            else:
                best_params = estimator.get_params()
                est_opt = estimator.fit(X_train, y_train)

                if verbose >= 2:

                    print('{modelname} (S{sim}/F{fold_count})'.format(modelname=modelname,
                                                                      sim=sim,
                                                                      fold_count=fold_count))

            y_pred = est_opt.predict(X_test)

            # record error metrics for fold x of sim y
            # y_tests.append(y_test)
            # y_preds.append(y_pred)







        # y_test = np.concatenate(y_tests, axis=0)
        # y_pred = np.concatenate(y_preds, axis=0)


            # accuracy, precision, auroc
            if outtype == 'cat':
                errs = {
                    'Acc' : np.round(accuracy_score(y_test, y_pred), 3),
                    'Prec' : np.round(precision_score(y_test, y_pred, average=None), 3),
                    'F1' : np.round(f1_score(y_test, y_pred, average=None), 3),
                    'Jaccard' : np.round(jaccard_score(y_test, y_pred, average=None), 3),
                }

            elif outtype == 'num':
                # Calculate adjusted R2 score
                r2 = r2_score(y_test, y_pred)
                n_samples = y.shape[0]
                n_predictors = X_test.shape[1]
                adj_r2 =  1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1))

                errs = {
                    'MAE' : np.round(mean_absolute_error(y_test, y_pred), 2),
                    'RMSE' : np.round(mean_squared_error(y_test, y_pred, squared=False), 2),
                    'R2' : np.round(r2, 3),
                    'Adj R2' : np.round(adj_r2, 3),
                }

            else:
                print('Error, outtype should be in [{outtypes}]'.format(outtypes=['cat', 'num']))

            if verbose >= 1:
                print('{modelname} (S{sim}/F{fold_count}) metrics: {errs}'.format(modelname=modelname,
                                                                                  sim=sim,
                                                                                  fold_count=fold_count,
                                                                                  errs=errs))

            fold_count += 1

            fold_output = [modelname, sim]

            for err_type in errs:
                fold_output.append(errs[err_type])

            err_list.append(fold_output)

    headers = ['Model', 'Sim']
    for err_type in errs:
        headers.append(err_type)

    err_df = pd.DataFrame(err_list, columns=headers)


    return err_df


# simulate n_sim imputations for a given estimator
def simulate_imputations(
    modelname,
    estimator,
    est_type,
    params,
    df,
    n_sim,
    n_sim_cv,
    n_gridsearch_cv,
    data_min_max,
    dist_type,
    rate,
    drop_max,
    size,
    noise,
    drop_proportion=1.0,            # proportion of indices in test set that have data missing
    crit_thresh=(10, 20, 30),
    output_ndigits=2,
    verbose=0,
    random_state=0,
):

    err_list = []

    if size is not None:
        train_prop = (n_sim_cv - 1)/n_sim_cv
        test_prop = 1/n_sim_cv

        # Need at least 1 datapoint in test group; if test proportion times size rounds down to 0,
        # increase it so that it rounds to 1, and proportionally set train proportion.
        if np.round(test_prop*size, 0) == 0:
            test_prop = 1/size
            train_prop = 1 - test_prop


        # If size is less than size_factor it is neccesary to repeat simulations to more accurately assess error
        # This is because low test-set sizes lead to wider variance in RMSE

        size_factor = 1280

        if size is None:
            size_multiplier = 1

        elif size < size_factor:
            size_multiplier = size_factor / size
            size_multiplier = int(np.round(size_multiplier, 0))

        else:
            size_multiplier = 1

    else:
        train_prop = None
        test_prop = None
        size_multiplier = 1


    for sim in range(0, n_sim):

        y_preds = []
        y_tests = []
        m_tests = []
        t_times = []
        p_times = []
        param_vals = []

        for size_iter in range(0, size_multiplier):

            kf = KFold(n_splits=n_sim_cv, shuffle=True)

            fold_count = 0
            for train_index, test_index in kf.split(df):
                fold_count += 1



                y_train, y_test = df.iloc[train_index], df.iloc[test_index]

                X_train, y_train, m_train = generate_sparse_dataset(
                    parent=y_train,
                    rate=rate,
                    dist_type=dist_type,
                    drop_proportion=drop_proportion,
                    drop_max=drop_max,
                    size=size,
                    prop=train_prop
                )

                X_test, y_test, m_test = generate_sparse_dataset(
                    parent=y_test,
                    rate=rate,
                    dist_type=dist_type,
                    drop_proportion=drop_proportion,
                    drop_max=drop_max,
                    size=size,
                    prop=test_prop
                )

                if (noise is not None) and (noise != 0):

                    noise_train = np.random.normal(loc=0.0, scale=noise/2, size=X_train.shape)
                    noise_test = np.random.normal(loc=0.0, scale=noise/2, size=X_test.shape)

                    X_train = X_train + noise_train
                    X_test = X_test + noise_test

                param_val = 0

                # If model is not NaN tolerant, remove NaN's from inputs - This ONLY works with custom drop-types
                if est_type == 'Model - not NaN tolerant':

                    if drop_dist != 'custom':
                        sys.exit('With models that are not NaN tolerant, drop_dist must be custom')

                    X_train = X_train.dropna(axis=1, how='all')
                    X_test = X_test.dropna(axis=1, how='all')

                    #est_op = estimator.fit(X_train, y_train)

                # Optimize hyperparameters
                start_time = time.process_time()

                # Interpolation
                if any(x in str(estimator) for x in ['Interpolator', 'LinearImputer', 'SplineImputer', 'PolynomialImputer']):
                    est_opt = estimator
                    #print('No optimization needed for imputation using ', model)

                # Mean, LinearImputer
                elif 'SimpleImputer' in str(estimator):
                    est_opt = estimator
                    best_params = estimator.get_params()
                    #print('No optimization needed for imputation using ', model)


                # MICE
                elif 'IterativeImputer' in str(estimator):

                    if len(params) > 0:
                        est_opt, best_params, mean_scores = imputer_gridsearch(estimator=estimator, X=X_train, y=y_train, params=params, n_gridsearch_cv=n_gridsearch_cv)
                        if verbose > 1: print('MICE best params: ', best_params)
                        if verbose > 2: print('MICE mean_scores: ', mean_scores)


                        #param_val = best_params
                    else:
                        best_params = estimator.get_params()
                        est_opt = estimator
                        #print('No param space defined, no optimization needed for MICE')



                elif 'MLP' in str(estimator):

                    X_train = X_train.fillna(-1)
                    X_test = X_test.fillna(-1)

                    if len(params) > 0:

                        est_opt = GridSearchCV(estimator=estimator,
                                                 param_grid=params,
                                                 cv=min(n_gridsearch_cv, X_train.shape[0]),
                                                 n_jobs=-1,
                                                 scoring='neg_mean_squared_error',
                                                 verbose=0)

                        est_opt = est_opt.fit(X_train, y_train)


                        if verbose > 1: print('MLP params: ', est_opt.best_params_)
                        if verbose > 2: print('GSoutput: ', est_opt.cv_results_['mean_test_score'])

                    else:
                        est_opt = estimator.fit(X_train, y_train)



                    est_opt = est_opt.fit(X_train, y_train)

                # KNN
                elif 'KNNImputer' in str(estimator):
                    if len(params) > 0:
                        est_opt, best_params, mean_scores = imputer_gridsearch(estimator=estimator, X=X_train, y=y_train, params=params, n_gridsearch_cv=n_gridsearch_cv)
                        if verbose > 1: print('KNN best params: ', best_params)
                        if verbose > 2: print('KNN mean_scores: ', mean_scores)
                        param_val = best_params['n_neighbors']
                    else:
                        best_params = estimator.get_params()
                        est_opt = estimator
                        #print('No param space defined, no optimization needed for KNNImputer')

                # MARS
                elif 'Earth' in str(estimator):

                    if len(params) > 0:
                        est_opt = GridSearchCV(estimator=estimator,
                                               param_grid=params,
                                               cv=min(n_gridsearch_cv, X_train.shape[0]),
                                               n_jobs=-1,
                                               scoring='neg_mean_squared_error',
                                               verbose=0)

                        est_opt = est_opt.fit(X_train, y_train)

                        #print('Time to optimize MARS: ', str(round((time.process_time() - start_time), 2)), ' - best params: ', est_opt.best_params_)

                        param_val = est_opt.best_params_['estimator__max_degree']
                    # If no param grid specified, fit input MARS model using default values
                    else:
                        est_opt = estimator
                        est_opt = est_opt.fit(X_train, y_train)
                        #print('Time to optimize MARS: ', str(round((time.process_time() - start_time), 2)), ' - best params: ', est_opt)

                # XGBoost
                elif 'XGBRegressor' in str(estimator):
                    estimator.estimator.random_state = np.random.randint(0, 9999)

                    if len(params) > 0:

                        est_opt = GridSearchCV(estimator=estimator,
                                                 param_grid=params,
                                                 cv=min(n_gridsearch_cv, X_train.shape[0]),
                                                 n_jobs=-1,
                                                 scoring='neg_mean_squared_error',
                                                 verbose=0)
                        est_opt = est_opt.fit(X_train, y_train)
                        if verbose > 2: print('XGB: ', est_opt.cv_results_['mean_test_score'])
                        #print('Time to optimize XGBoost: ', str(round((time.process_time() - start_time), 2)), ' - best params: ', est_opt.best_params_)

                    else:
                        est_opt = estimator
                        est_opt = est_opt.fit(X_train, y_train)
                        #print('Time to optimize XGBoost: ', str(round((time.process_time() - start_time), 2)), ' - best params: ', est_opt)

                else:
                    print('No valid estimator specified - estimator is: ', modelname)

                train_time = time.process_time() - start_time
                #
                # if verbose > 0:
                #     print('Model:'+str(model)+' - drops:'+str(drop_rate)+' - size:'+ str(size) + ' - noise:'+ str(noise)+
                #      ' - sim:'+str(sim+1), ' - size_iter:'+str(size_iter+1)+' - fold:'+str(fold_count)+' - train_time:'+str(train_time))


                # Use fitted estimator to generate predictions
                start_time = time.process_time()
                if est_type == 'Imputer':
                    # For imputers, we need to impute on the dataset as a whole, using y_train as inputs
                    n_test = X_test.shape[0]

                    # Add noise to y_train before concatenating to X_test
                    if (noise is not None) and (noise != 0):
                        noise_train = np.random.normal(loc=0.0, scale=noise/2, size=X_train.shape)
                        y_train = y_train + noise_train

                    X_test = pd.concat((X_test, y_train), axis=0)

                    est = estimator.set_params(**best_params)
                    y_pred = est.fit_transform(X_test)

                    #y_pred = est_opt.fit_transform(X_test)
                    y_pred = y_pred[0:n_test, :]

                    y_pred = np.clip(y_pred, a_min=data_min_max[0], a_max=data_min_max[1])

                elif est_type == 'Interpolator':
                    y_pred = estimator.fit_transform(X_test)
                    y_pred = np.clip(y_pred, a_min=data_min_max[0], a_max=data_min_max[1])

                elif est_type in ['Model', 'Model - not NaN tolerant']:
                    y_pred = est_opt.predict(X_test)
                    y_pred = np.clip(y_pred, a_min=data_min_max[0], a_max=data_min_max[1])

                else:
                    sys.exit((est_type, ' is not a valid est_type'))
                pred_time = time.process_time() - start_time

                print('Model:'+str(modelname)+'  Dist:'+str(dist_type)+'  Rate:'+str(rate)+'  Size:'+ str(size) + '  Noise:'+ str(noise)+
                      '  Sim:'+str(sim+1)+'  Fold:'+str(fold_count)+'  Ttime:'+str(train_time))


                y_preds.append(y_pred)
                y_tests.append(y_test)
                m_tests.append(m_test)
                t_times.append(train_time)
                p_times.append(pred_time)
                param_vals.append(param_val)

        y_preds = np.concatenate(y_preds, axis=0)
        y_tests = np.concatenate(y_tests, axis=0)
        m_tests = np.concatenate(m_tests, axis=0)
        t_times = mean(t_times)
        p_times = mean(p_times)
        param_vals = mean(param_vals)

                #print('lengths of y_preds is: ', len(y_preds))

        # Calculate errors
        sim_err = calc_error(y_pred=y_preds,
                             y_true=y_tests,
                             crit_thresh=crit_thresh,
                             train_time=t_times,
                             pred_time=p_times,
                             missing_mask=m_tests)

        # [Model, size, drop_rate, drop_dist, MAE, RMSE, R2, %EOT, traintime, testtime]

        sim_output = [modelname, dist_type, rate, noise, size]
        sim_mae = str(np.round(sim_err['Mean Absolute Error'], 3))
        sim_rmse = str(np.round(sim_err['Root Mean Squared Error'], 3))
        sim_r2 = str(np.round(sim_err['R2 Score'], 4))

        if verbose > 0:
            print('Simulation #{0} metrics  --  RMSE:{1}  MAE:{2}  R2:{3}'.format(str(sim), sim_rmse, sim_mae, sim_r2))

        for err_type in sim_err:
            sim_output.append(np.round(sim_err[err_type], 4))



        err_list.append(sim_output)



    headers = ['Model', 'Dist Type', 'Rate', 'Noise', 'Size']
    for err_type in sim_err:
        headers.append(err_type)

    err_df = pd.DataFrame(err_list, columns=headers)

    return err_df


def compare_estimators(
    X,
    y,
    outtype,
    estimators,
    n_sim,
    n_sim_cv,
    n_gridsearch_cv,
    gs_scoring,
    verbose=0
):

    model_errors = []

    for modelname in estimators:
        estimator = estimators[str(modelname)][0]
        params = estimators[str(modelname)][1]

        err_df_ = simulate_estimations(
            X=X,
            y=y,
            outtype=outtype,
            estimator=estimator,
            modelname=modelname,
            params=params,
            n_sim=n_sim,
            n_sim_cv=n_sim_cv,
            n_gridsearch_cv=n_gridsearch_cv,
            gs_scoring=gs_scoring,
            verbose=verbose
        )

        model_errors.append(err_df_)

    err_df = pd.concat(model_errors, ignore_index=True)

    return err_df


def compare_imputers(
    estimators,
    df,
    data_min_max,                 # (minimum data value, maximum data value)
    n_sim,
    n_sim_cv,
    n_gridsearch_cv,
    dist_types,           # 'random', 'parent', 'custom'
    rates,
    noises=[None],
    sizes=[None],
    drop_max=None,
    crit_thresh=(10, 20, 30),
    verbose=0,
    random_state=0,

):

    """
    Compares a performance for a list of estimators across a range of drop rates or dataset sizes

    Args:

        comparison_type (str): 'drop_rates' or 'sizes'

        comparison_range ([int]): Comprises the x axis of the output graphs. Defaults to None.
            If comparison_type is 'drop_rates', this represents the range of drop_rates tested.
            If comparison_type is 'sizes', this represents the range of dataset sizes tested.
    """

    model_errors = []

    # For each estimator we are comparing:
    for modelname in estimators:
        # Initialize list of errors - each index will represent another list of errors for a given drop_x
        #est_error = []
        estimator = estimators[str(modelname)][0]
        params = estimators[str(modelname)][1]

        # Imputation models
        if any(x in str(estimator) for x in [
            'KNNImputer',
            'SimpleImputer',
            'IterativeImputer'
        ]):
            est_type = 'Imputer'

        # Interpolation models
        elif any(x in str(estimator) for x in [
            'InstanceImputer',
            'LinearImputer',
            'SplineImputer',
            'PolynomialImputer',
            'Interpolator'
        ]):
            est_type = 'Interpolator'

        # For XGboost, MARS (Earth), LGBM, etc - estimators that are tolerant of missing values ...
        elif any(x in str(estimator) for x in [
            'XGBR',
            'Earth',
            'LGBM',
            'MLP'
        ]):
            est_type = 'Model'

        # Estimators that are not NaN-tolerant
        elif any(x in str(estimator) for x in [
            'RidgeCV',
            'LassoLarsCV',
            'SVR',
        ]):

            est_type = 'Model - not NaN tolerant'

        else:
            print(str(estimator))
            sys.exit('Need to include estimator', str(estimator), 'in decision tree')


        with ignore_warnings(category=(ConvergenceWarning,
                                       FutureWarning,
                                       RuntimeWarning)):


            for dist_type in dist_types:

                for rate in rates:

                    for size in sizes:

                        for noise in noises:

                            drop_x_err = simulate_imputations(
                                modelname=modelname,               # Name of model - a string
                                estimator=estimator,       # The actual estimator itself
                                est_type=est_type,
                                params=params,             # Param list for comparison
                                df=df,
                                data_min_max=data_min_max,
                                n_sim=n_sim,
                                n_sim_cv=n_sim_cv,
                                n_gridsearch_cv=n_gridsearch_cv,
                                dist_type=dist_type,
                                rate=rate,
                                size=size,
                                noise=noise,
                                drop_max=drop_max,
                                crit_thresh=crit_thresh,
                                verbose=verbose,
                                random_state=random_state
                            )

                            model_errors.append(drop_x_err)

        err_df = pd.concat(model_errors, ignore_index=True)

    return err_df

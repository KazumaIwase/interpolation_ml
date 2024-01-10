
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
import math
import time
import seaborn as sns
sns.set()


def time_split(train_x, train_y):
    """Returns training and validation dataset that are splited into three parts in chronological order"""
    start = 365*24
    stop = 1
    num = 4
    ind = list()
    for i in np.linspace(start, stop, num):
        ind.append(math.floor(i))

    tr_split = []
    va_split = []
    for i in range(3):
        if i == 2:
            tr_x, va_x = train_x.iloc[:-ind[i]], train_x.iloc[-ind[i]:]
            tr_y, va_y = train_y.iloc[:-ind[i]], train_y.iloc[-ind[i]:]
            tr_split.append((tr_x, tr_y))
            va_split.append((va_x, va_y))

        else:
            tr_x, va_x = train_x.iloc[:-ind[i]], train_x.iloc[-ind[i]:-ind[i+1]]
            tr_y, va_y = train_y.iloc[:-ind[i]], train_y.iloc[-ind[i]:-ind[i+1]]
            tr_split.append((tr_x, tr_y))
            va_split.append((va_x, va_y))
    return  tr_split, va_split


def get_scores(Model, tr_split, va_split, fixed_params, params):
    """Returns RMSE, and custom loss if you set metric parameter"""
    custom = ('objective' in fixed_params.keys())
    score_rmse = []
    if custom:
        score_custom = []
        custom_val = fixed_params['metric']

    for i in range(len(tr_split)):

        model = Model(fixed_params, params)
        model.fit(tr_split[i], va_split[i])

        y_pred = model.model.predict(va_split[i][0])
        score_rmse.append(((va_split[i][1] - y_pred)**2).mean()**0.5)
        if custom:
            score_custom.append(custom_val(y_pred, va_split[i][1] )[1])
    rmse = np.mean(score_rmse)
    if custom:
        val_loss = np.mean(score_custom)
        return rmse, val_loss
    else:
        return rmse


def plot_valid(y, pred, p, h, period=8*7*24, span=24*3, figsize=(18,6)):
    """Visualize results of validation data"""
    plt.figure(figsize=figsize)
    plt.plot(y.iloc[- period:].index, y.iloc[- period:], label = 'Target')
    plt.plot(y.iloc[- period:].index, pred[- period:], label = 'Prediction')
    plt.legend()
    if p == 'h':
        plt.title('Precipitation forecast for Hakone' + h)
        plt.ylabel('Precipitation(mm)')
        filename = 'Hakone' + h
    elif p == 'f':
        plt.title('Temperature forecast for Fujisan' + h)
        plt.ylabel('Temperature(℃)')
        filename = 'Fujisan' + h
    plt.xlabel('Time(hours)')
    plt.xticks([y.iloc[- period:].index[i] for i in np.arange(0, len(y.iloc[- period:].index), span)])
    plt.xticks(rotation=25)


def create_custom_loss(a=16.0, alpha=0.85):
    def custom_loss(y_pred, y_true):
        # compute the MSE loss
        if isinstance(y_true, lgb.basic.Dataset):
            y_true = y_true.get_label()
        mse_grad = y_pred - y_true
        mse_hess = np.ones_like(y_pred)

        # compute the new binary entropy loss
        y_true_class = np.where(y_true > 0, 1, 0)
        y_sigmoid =  1 / (1 + np.exp(- a * (y_pred - 0.25)))
        logloss_grad = a * (y_sigmoid - y_true_class)
        logloss_hess = a * a * y_sigmoid * (1.0 - y_sigmoid)

        # compute the combined gradient and Hessian
        beta = 1.0 - alpha # weight of LogLoss
        grad = alpha * mse_grad + beta * logloss_grad
        hess = alpha * mse_hess + beta * logloss_hess
        return grad, hess

    def custom_val(y_pred, y_true):
        # compute the MSE loss
        if isinstance(y_true, lgb.basic.Dataset):
            y_true = y_true.get_label()
        mse2 = np.mean((y_pred - y_true)**2)/2

        # compute the new binary entropy loss
        y_true_class = np.where(y_true > 0, 1, 0)
        y_sigmoid =  1 / (1 + np.exp(- a * (y_pred - 0.25)))
        logloss = np.mean( - y_true_class * np.log(y_sigmoid + 1e-10) - (1 - y_true_class) * np.log(1-y_sigmoid + 1e-10))

        # compute the root of combined loss
        beta = 1.0 - alpha # weight of LogLoss
        custom_val = alpha * mse2 + beta * logloss
        return 'custom_val', custom_val, False
    return custom_loss, custom_val



class Runner:
    """Provide various functions such as parameter tuning and training"""

    def __init__(self, Model, train_x, train_y):
        self.tr_split, self.va_split = time_split(train_x, train_y)
        self.best_params = None
        self.Model = Model
        self.train_x = train_x
        self.train_y = train_y
        self.best_score = None

    def run_opt(self, bayes_objective, fixed_params, n_trials=10, seed=42, round_num=3, show_history=False, figsize=(10,8), fontsize=30, linewidth=3):
        """Perform parameter tuning with optuna"""
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        if ('objective' in fixed_params.keys()) and (fixed_params['objective'] != 'regerssion'):
            history = {'val_loss':[], 'rmse':[]}
            objective = bayes_objective(self.Model, self.tr_split, self.va_split, fixed_params, history=history)
            study.optimize(objective, n_trials=n_trials)
            self.best_params = study.best_trial.params
            self.best_score = study.best_trial.value
            print(f'best params {self.best_params}\nbest score {round(self.best_score, round_num)}')

            if show_history:
                x = np.arange(len(study.trials))

                sort_index = np.argsort(history['val_loss'])[::-1]
                plt.figure(figsize=figsize, dpi=300)
                plt.plot(x, np.array(history['val_loss'])[sort_index], label='Binary Loss', linewidth=linewidth)
                plt.plot(x, np.array(history['rmse'])[sort_index], label='RMSE', linewidth=linewidth)
                plt.legend(fontsize=fontsize)
                plt.xlabel('Number of Optuna trials',fontsize=fontsize)
                plt.ylabel('Loss',fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

        else:
            objective = bayes_objective(self.Model, self.tr_split, self.va_split, fixed_params)
            study.optimize(objective, n_trials=n_trials)
            self.best_params = study.best_trial.params
            self.best_score = study.best_trial.value
            print(f'best params {self.best_params}\nbest score {round(self.best_score, round_num)}')

    def run_val(self, fixed_params, p, h, savefig=False):
        """See the validation results"""
        score_rmse=[]
        score_cv=[]
        for i in range(len(self.tr_split)):
            model = self.Model(fixed_params, self.best_params)
            model.fit(self.tr_split[i], self.va_split[i])
            self.run_model = model.model
            va_pred = self.run_model.predict(self.va_split[i][0])
            if ('objective' in fixed_params.keys()) and (fixed_params['objective'] != 'regerssion'):
                custom_val = fixed_params['metric']
                cv=custom_val(va_pred, self.va_split[i][1] )[1]
                print('Custom val loss', cv)
                score_cv.append(cv)
            rmse=((va_pred - self.va_split[i][1])**2).mean()**0.5
            print('RMSE', rmse)
            score_rmse.append(rmse)
            plot_valid(self.va_split[i][1], va_pred, p=p, h=h)
        print('RMSE avg', np.mean(score_rmse))
        if ('objective' in fixed_params.keys()) and (fixed_params['objective'] != 'regerssion'):
            print('Custom val loss avg', np.mean(score_cv))

    def run_importanace(self, fixed_params, figsize=(10,8), fontsize=20, linewidth=3, nf=False):
        """Compute important features"""
        fi = self.run_model.feature_importance(importance_type='split')
        idx = np.argsort(fi)[::-1]
        self.cols, self.importances = self.train_x.columns.values[idx], fi[idx]

        if nf:
            rmse=[]
            con=[]
            if 'sin_hour' in self.train_x.columns:
                sc = np.array(['sin_day', 'cos_day', 'sin_hour', 'cos_hour'])
            else:
                sc = np.array(['sin_day', 'cos_day'])
            cols=self.cols
            for i in sc:
                cols=cols[cols != i]
            for j in np.arange(10,len(self.cols),10):
                top_cols = np.union1d(sc, cols[:j-len(sc)])
                tr_split, va_split = time_split(self.train_x[top_cols], self.train_y)
                rmse.append(get_scores(self.Model, tr_split, va_split, fixed_params, self.best_params))
                con.append(j)

            self.top=con[np.argmin(np.array(rmse))]
            print(f'top{self.top}')
            #self.top_cols, top_importances = self.cols[:self.top], self.importances[:self.top]

            plt.figure(figsize=figsize, dpi=300)
            plt.plot(con, rmse, linewidth=linewidth)
            plt.xlabel('Number of Features',fontsize=fontsize)
            plt.ylabel('RMSE',fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

    def get_cols(self, top=70):
        """Return the important features"""

        if 'sin_hour' in self.train_x.columns:
            sc = np.array(['sin_day', 'cos_day', 'sin_hour', 'cos_hour'])
        else:
            sc = np.array(['sin_day', 'cos_day'])

        if top > len(self.cols):
            top = len(self.cols)

        cols=self.cols
        for i in sc:
            cols=cols[cols != i]
        top_cols = np.union1d(sc, cols[:top-len(sc)])
        return top_cols

    def plot_importanace(self, figsize=(10,8), top=10, fontsize=20):
        top_cols, top_importances = self.cols[:top], self.importances[:top]

        importances = pd.Series(top_importances, index=top_cols).sort_values(ascending=True)
        plt.figure(figsize=figsize, dpi=300)
        importances.plot.barh(fontsize=fontsize)

    def run_train_all(self, fixed_params, eval_metric='rmse'):
        """Training on all training data"""
        if 'n_estimators' in fixed_params:
            fixed_params['n_estimators'] = self.run_model.best_iteration
            print('best_iteration', self.run_model.best_iteration)
        elif 'nb_epoch' in fixed_params:
            fixed_params['nb_epoch'] = self.run_model.best_iteration
            print('best_iteration', self.run_model.best_iteration)
        model = self.Model(fixed_params, self.best_params)
        t_start = time.time()
        model.fit([self.train_x, self.train_y])
        t_end = time.time()
        self.t_train = t_end - t_start
        print(self.t_train)
        self.model = model.model


class TestRun:
    """Provides various processing for test data"""
    def __init__(self, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y
        self.y_news = None
        self.f31 = None
        self.f44 = None
        self.df_rmse = None

    def runf_test(self, model, test_set, f3144, top_cols, ts_time, loc='Temperature_Fujisan', round_num=3):
        """Compute RMSE of Mt.Fuji for the prediction, Weather News and Tenki Tokurasu"""
        self.ts_lgbm = model.predict(self.test_x[top_cols])
        self.y_news = test_set.at_time(ts_time).loc[:, loc]
        self.f31 = f3144.at_time(ts_time)['f31']
        self.f44 = f3144.at_time(ts_time)['f44']
        score_lg = mean_squared_error(self.test_y, self.ts_lgbm , squared=False)
        score_news = mean_squared_error(self.test_y, self.y_news, squared=False)
        # score_31 = mean_squared_error(self.test_y, self.f31, squared=False)
        # score_44 = mean_squared_error(self.test_y, self.f44, squared=False)
        score_3144 = mean_squared_error(self.test_y,
             (4400-3775)/(4400-3100) * self.f31 + (3775-3100)/(4400-3100) * self.f44,
             squared=False)
        self.df_rmse = pd.DataFrame({'LightGBM': [round(score_lg, round_num)],
                                     'Weathernews': [round(score_news, round_num)],
                                     #'Tenki(3100m)': [round(score_31, round_num)],
                                     #'Tenki(4400m)': [round(score_44, round_num)],
                                     'Tenki(3775m)': [round(score_3144, round_num)]},
                                     index=['RMSE'], columns=['LightGBM', 'Weathernews',
                                     #'Tenki(3100m)', 'Tenki(4400m)'
                                     'Tenki(3775m)'])

    def runf_test_without_f3144(self, model, test_set, top_cols, ts_time, loc='Temperature_Fujisan', round_num=3):
        """Compute RMSE of Mt.Fuji for the prediction and Weather News"""
        self.ts_lgbm = model.predict(self.test_x[top_cols])
        self.y_news = test_set.at_time(ts_time).loc[:, loc]
        score_lg = mean_squared_error(self.test_y, self.ts_lgbm , squared=False)
        score_news = mean_squared_error(self.test_y, self.y_news, squared=False)
        self.df_rmse = pd.DataFrame({'LightGBM': [round(score_lg, round_num)],
                                     'Weathernews': [round(score_news, round_num)]},
                                     index=['RMSE'], columns=['LightGBM', 'Weathernews'])

    def runh_test(self, model, test_set, top_cols, ts_time, loc='Precipitation_Hakone', round_num=3):
        """Compute RMSE of Hakone for the prediction and Weather News"""
        self.ts_lgbm = model.predict(self.test_x[top_cols])
        self.ts_lgbm_threshold = self.ts_lgbm * (self.ts_lgbm > 0.3) + 0.0 * (self.ts_lgbm <= 0.3)
        self.y_news = test_set.at_time(ts_time).loc[:, loc]
        score_lg = mean_squared_error(self.test_y, self.ts_lgbm , squared=False)
        score_lg_threshold = mean_squared_error(self.test_y, self.ts_lgbm_threshold , squared=False)
        score_news = mean_squared_error(self.test_y, self.y_news, squared=False)
        self.df_rmse = pd.DataFrame({'LightGBM': [round(score_lg, round_num)],
                                     'LightGBM_threshold': [round(score_lg_threshold, round_num)],
                                     'Weathernews': [round(score_news, round_num)]},
                                     index=['RMSE'], columns=['LightGBM', 'LightGBM_threshold', 'Weathernews'])

    def plot_test(self, ylabel, s, title=False, figsize=(22,10), fontsize=10, linewidth=3, skip=20, threshold=False):
        """Visualize the results of predictions along with forecasts and observations"""
        plt.figure(figsize=figsize, dpi=300)
        plt.plot(self.test_y.index, self.test_y, label = 'Observation', linewidth=linewidth)
        if threshold:
            plt.plot(self.test_y.index, self.ts_lgbm_threshold , label = 'LightGBM_threshold', linewidth=linewidth)
        else:
            plt.plot(self.test_y.index, self.ts_lgbm , label = 'LightGBM', linewidth=linewidth)
        if s == 'w':
            plt.plot(self.test_y.index, self.y_news, label = 'Weathernews', linewidth=linewidth)
        elif s == '3':
            plt.plot(self.test_y.index, self.f31, label = 'Tenki(3100m)', linewidth=linewidth)
        elif s == '4':
            plt.plot(self.test_y.index, self.f44, label = 'Tenki(4400m)', linewidth=linewidth)
        elif s == '34':
            plt.plot(self.test_y.index,
                (4400-3775)/(4400-3100) * self.f31 + (3775-3100)/(4400-3100) * self.f44,
                label = 'Tenki(3775m)', linewidth=linewidth)
        plt.legend(fontsize=fontsize)
        if title is not False:
            plt.title(title)
        plt.ylabel(ylabel,fontsize=fontsize)
        plt.xticks([self.test_y.index[i] for i in np.arange(0, len(self.test_y.index), skip)],fontsize=fontsize,rotation=25)
        plt.yticks(fontsize=fontsize)

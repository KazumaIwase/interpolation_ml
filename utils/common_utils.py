import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
import math
import seaborn as sns
sns.set()


def time_split(train_x, train_y):
    """Returns training and validation dataset that are splited into three parts in chronological order."""
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


def plot_valid(y, pred, p, h, period=8*7*24, span=24*3, figsize=(18,6), savefig=False):
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
    if savefig:
        plt.savefig(filename + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')
    plt.show()


def create_custom_loss(a=16.0, alpha=0.85):
    def custom_loss(y_pred, y_true):
        # compute the RMSE loss
        if isinstance(y_true, lgb.basic.Dataset):
            y_true = y_true.get_label()
        rmse_grad = y_pred - y_true
        rmse_hess = np.ones_like(y_pred)

        # compute the new binary entropy loss
        y_true_class = np.where(y_true > 0, 1, 0)
        y_sigmoid =  1 / (1 + np.exp(- a * (y_pred - 0.25)))
        logloss_grad = a * (y_sigmoid - y_true_class)
        logloss_hess = a * a * y_sigmoid * (1.0 - y_sigmoid)

        # compute the combined gradient and Hessian
        beta = 1.0 - alpha # weight of LogLoss
        grad = alpha * rmse_grad + beta * logloss_grad
        hess = alpha * rmse_hess + beta * logloss_hess
        return grad, hess

    def custom_val(y_pred, y_true):
        # compute the RMSE loss
        if isinstance(y_true, lgb.basic.Dataset):
            y_true = y_true.get_label()
        rmse = np.mean((y_pred - y_true)**2)**0.5

        # compute the new binary entropy loss
        y_true_class = np.where(y_true > 0, 1, 0)
        y_sigmoid =  1 / (1 + np.exp(- a * (y_pred - 0.25)))
        logloss = np.mean( - y_true_class * np.log(y_sigmoid + 1e-10) - (1 - y_true_class) * np.log(1-y_sigmoid + 1e-10))

        # compute the combined gradient and Hessian
        beta = 1.0 - alpha # weight of LogLoss
        custom_val = alpha * rmse + beta * logloss
        return 'custom_val', custom_val, False
    return custom_loss, custom_val


class Runner:
    def __init__(self, Model, train_x, train_y):
        self.tr_split, self.va_split = time_split(train_x, train_y)
        self.best_params = None
        self.Model = Model
        self.train_x = train_x
        self.train_y = train_y
        self.top_cols = None
        self.best_score = None

    def run_opt(self, bayes_objective, fixed_params, n_trials=10, seed=42, round_num=3, show_history=True):
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
                plt.plot(x, np.array(history['val_loss'])[sort_index], label='custom_loss')
                plt.plot(x, np.array(history['rmse'])[sort_index], label='rmse')
                plt.legend()
                plt.title('Sorted validation loss')
                plt.show()
        else:
            objective = bayes_objective(self.Model, self.tr_split, self.va_split, fixed_params)
            study.optimize(objective, n_trials=n_trials)
            self.best_params = study.best_trial.params
            self.best_score = study.best_trial.value
            print(f'best params {self.best_params}\nbest score {round(self.best_score, round_num)}')

    def run_val(self, fixed_params, p, h, savefig=False):
        for i in range(len(self.tr_split)):
            model = self.Model(fixed_params, self.best_params)
            model.fit(self.tr_split[i], self.va_split[i])
            self.run_model = model.model
            va_pred = self.run_model.predict(self.va_split[i][0])
            if ('objective' in fixed_params.keys()) and (fixed_params['objective'] != 'regerssion'):
                custom_val = fixed_params['metric']
                print('Custom val loss', custom_val(va_pred, self.va_split[i][1] )[1])
            print('RMSE', ((va_pred - self.va_split[i][1])**2).mean()**0.5)
            plot_valid(self.va_split[i][1], va_pred, p=p, h=h, savefig=savefig)

    def run_importanace(self, title='Feature importance', figsize=(6,15), top=70):
        fi = self.run_model.feature_importance(importance_type='split')
        idx = np.argsort(fi)[::-1]
        self.top_cols, top_importances = self.train_x.columns.values[idx][:top], fi[idx][:top]

        importances = pd.Series(top_importances, index=self.top_cols).sort_values(ascending=True)
        plt.figure(figsize=figsize)
        importances.plot.barh()
        plt.title(title)
        plt.show()

    def run_train_all(self, fixed_params, eval_metric='rmse'):
        if 'n_estimators' in fixed_params:
            fixed_params['n_estimators'] = self.run_model.best_iteration
        elif 'nb_epoch' in fixed_params:
            fixed_params['nb_epoch'] = self.run_model.best_iteration
        model = self.Model(fixed_params, self.best_params)
        model.fit([self.train_x, self.train_y])
        self.model = model.model


class TestRun:
    def __init__(self, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y
        self.y_news = None
        self.f31 = None
        self.f44 = None
        self.df_rmse = None

    def runf_test(self, model, test_set, f3144, top_cols, ts_time, loc='Temperature_Fujisan', round_num=3):
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
        self.ts_lgbm = model.predict(self.test_x[top_cols])
        self.y_news = test_set.at_time(ts_time).loc[:, loc]
        score_lg = mean_squared_error(self.test_y, self.ts_lgbm , squared=False)
        score_news = mean_squared_error(self.test_y, self.y_news, squared=False)
        self.df_rmse = pd.DataFrame({'LightGBM': [round(score_lg, round_num)],
                                     'Weathernews': [round(score_news, round_num)]},
                                     index=['RMSE'], columns=['LightGBM', 'Weathernews'])

    def runh_test(self, model, test_set, top_cols, ts_time, loc='Precipitation_Hakone', round_num=3):
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

    def plot_test(self, title, ylabel, s, figsize=(16,4), skip=20, threshold=False):
        plt.figure(figsize=figsize)
        plt.plot(self.test_y.index, self.test_y, label = 'Target')
        if threshold:
            plt.plot(self.test_y.index, self.ts_lgbm_threshold , label = 'LightGBM_threshold')
        else:
            plt.plot(self.test_y.index, self.ts_lgbm , label = 'LightGBM')
        if s == 'w':
            plt.plot(self.test_y.index, self.y_news, label = 'Weathernews')
        elif s == '3':
            plt.plot(self.test_y.index, self.f31, label = 'Tenki(3100m)')
        elif s == '4':
            plt.plot(self.test_y.index, self.f44, label = 'Tenki(4400m)')
        elif s == '34':
            plt.plot(self.test_y.index,
                (4400-3775)/(4400-3100) * self.f31 + (3775-3100)/(4400-3100) * self.f44,
                label = 'Tenki(3775m)')
        plt.legend()
        if title is not False:
            plt.title(title)
        # plt.xlabel('Time(hours)')
        plt.ylabel(ylabel)
        plt.xticks([self.test_y.index[i] for i in np.arange(0, len(self.test_y.index), skip)])
        plt.xticks(rotation=25)
        if title is not False:
            plt.savefig(title + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')
        else:
            plt.savefig(ylabel + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')
        plt.show()
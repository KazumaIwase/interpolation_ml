import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
import optuna
import time
import math
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
    scores = []
    for i in range(len(tr_split)):
        va_x, va_y = va_split[i]
        model = Model(fixed_params, params)
        model.fit(tr_split[i], va_split[i])
        va_pred = model.predict(va_x)
        score = mean_squared_error(va_y, va_pred, squared=False)
        scores.append(score)
        val = np.mean(scores)
    return val


def plot_valid(y, pred, p, h, period=8*7*24, span=24*3, figsize=(18,6)):
    """Visualize results of validation data"""
    plt.figure(figsize=figsize)
    plt.plot(y.iloc[- period:].index, y.iloc[- period:], label = 'Target')
    plt.plot(y.iloc[- period:].index, pred[- period:], label = 'Prediction')
    plt.legend()
    if p == 'h':
        plt.title('Precipitation forecast for Hakone' + h)
        plt.ylabel('Precipitation(mm)')
    elif p == 'f':
        plt.title('Temperature forecast for Fujisan' + h)
        plt.ylabel('Temperature(℃)')
    plt.xlabel('Time(hours)')
    plt.xticks([y.iloc[- period:].index[i] for i in np.arange(0, len(y.iloc[- period:].index), span)])
    plt.xticks(rotation=25)
    plt.show()


class Runner:
    """Provide various functions such as parameter tuning and training"""

    def __init__(self, Model, train_x, train_y):
        self.tr_split, self.va_split = time_split(train_x, train_y)
        self.best_params = None
        self.Model = Model
        self.model = None
        self.run_model = None
        self.train_x = train_x
        self.train_y = train_y
        self.best_iteration_ = None
        self.top_cols = None
        self.best_score = None

    def run_opt(self, bayes_objective, fixed_params, n_trials=50, seed=42, round_num=3):
        """Perform parameter tuning with optuna"""
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(bayes_objective(Model=self.Model, tr_split=self.tr_split, va_split=self.va_split, fixed_params=fixed_params), n_trials = n_trials, n_jobs = 1)

        self.best_params = study.best_trial.params
        self.best_score = study.best_trial.value
        print(f'best params {self.best_params}\nbest score {round(self.best_score, round_num)}')

    def hand_opt(self, params, fixed_params, round_num=3):
        """Perform manual parameter tuning"""
        self.best_params = params
        self.score = get_scores(self.Model, self.tr_split, self.va_split, fixed_params, params)
        print(f'params {self.best_params}\nscore {round(self.score, round_num)}')

    def run_val(self, fixed_params, p, h):
        """See the validation results"""
        self.run_model = self.Model(fixed_params, self.best_params)
        self.run_model.fit(self.tr_split[-1], self.va_split[-1])
        va_pred = self.run_model.predict(self.va_split[-1][0])
        plot_valid(self.va_split[-1][1], va_pred, p=p, h=h)
        self.val_score = mean_squared_error(self.va_split[-1][1], va_pred, squared=False)

    def run_importanace(self, title='Feature importance', figsize=(6,15), top=70):
        """Compute important features"""
        fi = self.run_model.feature_importances_
        idx = np.argsort(fi)[::-1]
        self.top_cols, top_importances = self.train_x.columns.values[idx][:top], fi[idx][:top]

        importances = pd.Series(top_importances, index=self.top_cols).sort_values(ascending=True)
        plt.figure(figsize=figsize)
        importances.plot.barh()
        plt.title(title)
        plt.show()

    def plot_importanace(self, figsize=(3,6), top=10):
        fi = self.run_model.feature_importances_
        idx = np.argsort(fi)[::-1]
        top_cols, top_importances = self.train_x.columns.values[idx][:top], fi[idx][:top]

        importances = pd.Series(top_importances, index=top_cols).sort_values(ascending=True)
        plt.figure(figsize=figsize)
        importances.plot.barh()
        plt.show()

    def run_curve(self):
        """Visualize loss trends"""
        loss = self.run_model.loss
        va_loss = self.run_model.va_loss
        plt.plot(np.arange(len(loss)) + 0.5, loss, 'b.-', label='Training loss')
        plt.plot(np.arange(len(va_loss)) + 1, va_loss, 'r.-', label='Validation loss')
        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        plt.legend(fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def run_train_all(self, fixed_params):
        """Training on all training data"""
        t_start = time.time()

        if 'n_estimators' in fixed_params:
            fixed_params['n_estimators'] = self.run_model.best_iteration_
        elif 'nb_epoch' in fixed_params:
            fixed_params['nb_epoch'] = self.run_model.best_iteration_
        self.model = self.Model(fixed_params, self.best_params)
        self.model.fit((self.train_x, self.train_y))

        t_end = time.time()
        self.t_train = t_end - t_start 


def val_rmse(scores, title, figsize=(15, 5), round_num=3, round_time=3):
    """Returns validation RMSE and training time of models"""
    df_plot = pd.DataFrame([{key: scores[key][0] for key in scores}, {key: scores[key][1] for key in scores}],
                            index=['RMSE', 'Training time(s)'], columns=[key for key in scores]).T
    axes = df_plot.plot.bar(title=title, figsize=figsize, rot=0, subplots=True, legend=False)
    axes[0].set_ylabel('RMSE')
    axes[1].set_ylabel('Training time(s)')

    df_models = pd.DataFrame([{key: round(scores[key][0], round_num) for key in scores}, {key: round(scores[key][1], round_time) for key in scores}],
                            index=['RMSE', 'Training time(s)'], columns=[key for key in scores])
    return df_models
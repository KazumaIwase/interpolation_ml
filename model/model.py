import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler


# elastic(linear)
class Elastic:
    def __init__(self, fixed_params, params):
        self.fixed_params = fixed_params
        self.params = params
        self.scaler = None
        self.model = None

    def fit(self, tr_split, va_split=None):
        seed = self.fixed_params['seed']

        self.scaler = StandardScaler()
        tr_x = self.scaler.fit_transform(tr_split[0])
        tr_y = tr_split[1]

        self.model = ElasticNet(random_state=seed)
        self.model.set_params(**self.params)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred


# lightgbm(Scikit-learn API)
class LgbmSL:
    def __init__(self, fixed_params, params):
        self.fixed_params = fixed_params
        self.params = params
        self.model = None
        self.best_iteration_ = None
        self.feature_importances_ = None

    def fit(self, tr_split, va_split=None):
        patience = self.fixed_params['patience']
        verbose = self.fixed_params['verbose']
        n_estimators = self.fixed_params['n_estimators']

        self.model = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                                       n_estimators=n_estimators, n_jobs = -1, random_state=42)
        self.model.set_params(**self.params)

        if patience == False:
            self.model.fit(tr_split[0], tr_split[1], eval_metric='rmse', verbose=verbose)
        else:
            self.model.fit(tr_split[0], tr_split[1], eval_metric='rmse', eval_set=[va_split], verbose=verbose,
                           early_stopping_rounds=patience)
            self.best_iteration_ = self.model.best_iteration_
            self.feature_importances_ = self.model.booster_.feature_importance(importance_type='split')

    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred


# lightgbm(Training API)
class Lgbm:
    def __init__(self, fixed_params, params):
        self.fixed_params = fixed_params
        self.params = params
        self.model = None

    def fit(self, tr_split, va_split=None):

        if ('objective' in self.fixed_params.keys()) and (self.fixed_params['objective'] != 'regerssion'):
            self.params['metric'] = None
            fobj = self.fixed_params['objective']
            feval = self.fixed_params['metric']
        else:
            self.params['metric'] = 'rmse'
            fobj = None
            feval = None

        if 'verbose_eval' in self.fixed_params.keys():
            verbose_eval = self.fixed_params['verbose_eval']
        else:
            verbose_eval = False

        n_estimators = self.fixed_params['n_estimators']

        if 'patience' in self.fixed_params.keys():
            if self.fixed_params['patience'] == False:
                callbacks = []
            else:
                callbacks=[lgb.early_stopping(stopping_rounds=self.fixed_params['patience'],
                                              verbose=False),
                          ]
        else:
      			callbacks = []

        self.params['verbosity'] = -1

        data_train = lgb.Dataset(tr_split[0], tr_split[1])
        if va_split == None:
            self.model = lgb.train(self.params, data_train, valid_sets = None,
                                  num_boost_round = n_estimators,
                                  fobj=fobj, feval=feval,
                                  verbose_eval=verbose_eval)
        else:
            data_val = lgb.Dataset(va_split[0], va_split[1])
            self.model = lgb.train(self.params, data_train, valid_sets = data_val,
                                  num_boost_round = n_estimators,
                                  fobj=fobj, feval=feval, callbacks=callbacks,
                                  verbose_eval=verbose_eval)


# xgboost
class Xgboost:
    def __init__(self, fixed_params, params):
        self.fixed_params = fixed_params
        self.params = params
        self.model = None
        self.best_iteration_ = None

    def fit(self, tr_split, va_split=None):
        patience = self.fixed_params['patience']
        verbose = self.fixed_params['verbose']
        n_estimators = self.fixed_params['n_estimators']

        self.model = xgb.XGBRegressor(booster='gbtree', objective='reg:squarederror', n_jobs = -1,
                                      random_state=42, n_estimators= n_estimators, learning_late = 0.1)
        self.model.set_params(**self.params)

        if patience == False:
            self.model.fit(tr_split[0], tr_split[1], eval_metric='rmse', verbose=verbose)
        else:
            self.model.fit(tr_split[0], tr_split[1], eval_metric='rmse', eval_set=[va_split], verbose=verbose,
                            early_stopping_rounds=patience)
            self.best_iteration_ = self.model.best_iteration

    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred


# ExtraTreesRegressor
class ExtraTrees:
    def __init__(self, fixed_params, params):
        self.fixed_params = fixed_params
        self.params = params
        self.model = None

    def fit(self, tr_split, va_split=None):
        self.model = ExtraTreesRegressor(random_state=42, n_jobs = -1)
        self.model.set_params(**self.params)
        self.model.fit(tr_split[0], tr_split[1])

    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred


# neural net
class MLP:
    def __init__(self, fixed_params, params):
        self.fixed_params = fixed_params
        self.params = params
        self.scaler = None
        self.model = None
        self.history = None
        self.best_iteration_ = None
        self.loss = None
        self.va_loss = None

    def fit(self, tr_split, va_split=None):
        # base parameters
        patience = self.fixed_params['patience']
        verbose = self.fixed_params['verbose']
        learning_rate = self.fixed_params['learning_rate']
        nb_epoch = self.fixed_params['nb_epoch']

        # parameters
        hidden_units1 = self.params['hidden_units1']
        hidden_units2 = self.params['hidden_units2']
        hidden_units3 = self.params['hidden_units3']
        hidden_dropout = self.params['hidden_dropout']

        self.model = tf.keras.models.Sequential()

        # input layer
        self.model.add(tf.keras.Input(shape=(tr_split[0].shape[1:])))

        # hidden layer
        for hidden_units in [hidden_units1, hidden_units2, hidden_units3]:
            self.model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(hidden_dropout))

        # output layer
        self.model.add(tf.keras.layers.Dense(1))

        # setting an objective function and metics
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                           metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        batch_size = 64
        if patience == False:
            # standardization
            self.scaler = StandardScaler()
            tr_x = self.scaler.fit_transform(tr_split[0])
            self.history = self.model.fit(tr_x, tr_split[1], workers = -1, use_multiprocessing = True,
                                          epochs=nb_epoch, verbose=verbose, batch_size = batch_size)
        else:
            # standardization
            self.scaler = StandardScaler()
            tr_x = self.scaler.fit_transform(tr_split[0])
            va_x = self.scaler.transform(va_split[0])
            va_split = (va_x, va_split[1])

            early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)

            self.history = self.model.fit(tr_x, tr_split[1], workers = -1, use_multiprocessing = True,
                                          epochs=nb_epoch, verbose=verbose, batch_size = batch_size,
                                          validation_data=va_split, callbacks=[early_stopping])
            self.best_iteration_ = len(self.history.history['loss'])
            self.loss = self.history.history['loss']
            self.va_loss = self.history.history['val_loss']


    def predict(self, x):
        # prediction
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred
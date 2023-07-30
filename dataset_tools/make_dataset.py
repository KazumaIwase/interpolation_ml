import pandas as pd
import numpy as np


def get_test(test_y, start_time, dataset, get_data, news_data):
    """Returns test dataset from observation data and forecast data."""
    for i in range(len(test_y)):
        dataset_box = dataset.copy().loc['2022-09-01 09:00:00':]
        start = start_time.index.values[i]
        end = test_y.index.values[i]

        dataset_box.loc[start:end,:] = news_data.loc[start:end:] # changes only from the present time to the predicting time
        test_x_all, _ = get_data(dataset_box)

        if i ==0:
            test_data = test_x_all.loc[end]
        else:
            test_data = pd.concat([test_data, test_x_all.loc[end]], axis=1)
    return test_data.T


def make_exp_dataset(x,f):
    x_all, y = f(x)
    train_x_all, train_y = x_all.loc[:'2022-10-16 07:00:00'], y.loc[:'2022-10-16 07:00:00']
    return train_x_all, train_y


def make_dataset(x,n,f,h):
    x_all, y = f(x)
    train_x_all, train_y = x_all.loc[:'2022-10-16 07:00:00'], y.loc[:'2022-10-16 07:00:00']
    test_y_all = y.loc['2022-10-16 08:00:00':]
    test_y = test_y_all.at_time(h)

    start_time = test_y_all.at_time('08:00:00')
    test_x = get_test(test_y, start_time, x, f, n)
    return train_x_all, train_y, test_x, test_y


def make_no_future_dataset(x,f,h):
    x_all, y = f(x)
    train_x_all, train_y = x_all.loc[:'2022-10-16 07:00:00'], y.loc[:'2022-10-16 07:00:00']
    test_y_all = y.loc['2022-10-16 08:00:00':]

    test_y = test_y_all.at_time(h)
    test_x = x_all.loc[test_y.index.values]
    return train_x_all, train_y, test_x, test_y


def get_data_mtfuji_2h(dataset):
    df = pd.concat([
                    pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.sin(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1),
                    dataset.shift(2),
                    dataset.shift(3),
                    dataset.shift(4),
                    dataset.shift(5),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0).diff(),
                    dataset.shift(2).diff(),
                    dataset.shift(3).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day', 'sin_hour', 'cos_hour'] \
                + [i + '_t' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-2' for i in dataset.columns] \
                + [i + '_t-3' for i in dataset.columns] \
                + [i + '_t-4' for i in dataset.columns] \
                + [i + '_t-5' for i in dataset.columns] \
                + [i + 'diff0' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff2' for i in dataset.columns] \
                + [i + 'diff3' for i in dataset.columns] \
                + [i + '_mean_t-2-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-2-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns]
    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Temperature_Fujisan']

    return train_x, train_y


def get_data_mtfuji_7h(dataset):
    df = pd.concat([
                    pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.sin(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4),
                    dataset.shift(7),
                    dataset.shift(8),
                    dataset.shift(9),
                    dataset.shift(23),
                    dataset.shift(24),
                    dataset.shift(25),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2).diff(),
                    dataset.shift(7).diff(),
                    dataset.shift(8).diff(),
                    dataset.shift(24).diff(),
                    dataset.shift(7).diff(periods=24),
                    dataset.shift(7).diff(periods=24*7),
                    dataset.shift(24).diff(periods=24),
                    dataset.shift(24).diff(periods=24*7),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day', 'sin_hour', 'cos_hour'] \
                + [i + '_t' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-7' for i in dataset.columns] \
                + [i + '_t-8' for i in dataset.columns] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + '_t-23' for i in dataset.columns] \
                + [i + '_t-24' for i in dataset.columns] \
                + [i + '_t-25' for i in dataset.columns] \
                + [i + 'diff0' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff7' for i in dataset.columns] \
                + [i + 'diff8' for i in dataset.columns] \
                + [i + 'diff24' for i in dataset.columns] \
                + [i + 'diff7+24' for i in dataset.columns] \
                + [i + 'diff7+24*7' for i in dataset.columns] \
                + [i + 'diff24+24' for i in dataset.columns] \
                + [i + 'diff24+24*7' for i in dataset.columns] \
                + [i + '_mean_t-7-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-7-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns]
    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Temperature_Fujisan']

    return train_x, train_y


def get_data_mtfuji_8h(dataset):
    df = pd.concat([
                    pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.sin(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4),
                    dataset.shift(8),
                    dataset.shift(9),
                    dataset.shift(10),
                    dataset.shift(23),
                    dataset.shift(24),
                    dataset.shift(25),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2).diff(),
                    dataset.shift(8).diff(),
                    dataset.shift(9).diff(),
                    dataset.shift(24).diff(),
                    dataset.shift(8).diff(periods=24),
                    dataset.shift(8).diff(periods=24*7),
                    dataset.shift(24).diff(periods=24),
                    dataset.shift(24).diff(periods=24*7),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day', 'sin_hour', 'cos_hour'] \
                + [i + '_t' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-8' for i in dataset.columns] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + '_t-10' for i in dataset.columns] \
                + [i + '_t-23' for i in dataset.columns] \
                + [i + '_t-24' for i in dataset.columns] \
                + [i + '_t-25' for i in dataset.columns] \
                + [i + 'diff0' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff8' for i in dataset.columns] \
                + [i + 'diff9' for i in dataset.columns] \
                + [i + 'diff24' for i in dataset.columns] \
                + [i + 'diff8+24' for i in dataset.columns] \
                + [i + 'diff8+24*7' for i in dataset.columns] \
                + [i + 'diff24+24' for i in dataset.columns] \
                + [i + 'diff24+24*7' for i in dataset.columns] \
                + [i + '_mean_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns]
    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Temperature_Fujisan']

    return train_x, train_y


def get_data_mtfuji_9h(dataset):
    df = pd.concat([
                    pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.sin(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4),
                    dataset.shift(9),
                    dataset.shift(10),
                    dataset.shift(11),
                    dataset.shift(23),
                    dataset.shift(24),
                    dataset.shift(25),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2).diff(),
                    dataset.shift(9).diff(),
                    dataset.shift(10).diff(),
                    dataset.shift(24).diff(),
                    dataset.shift(9).diff(periods=24),
                    dataset.shift(9).diff(periods=24*7),
                    dataset.shift(24).diff(periods=24),
                    dataset.shift(24).diff(periods=24*7),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day', 'sin_hour', 'cos_hour'] \
                + [i + '_t' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + '_t-10' for i in dataset.columns] \
                + [i + '_t-11' for i in dataset.columns] \
                + [i + '_t-23' for i in dataset.columns] \
                + [i + '_t-24' for i in dataset.columns] \
                + [i + '_t-25' for i in dataset.columns] \
                + [i + 'diff0' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff9' for i in dataset.columns] \
                + [i + 'diff10' for i in dataset.columns] \
                + [i + 'diff24' for i in dataset.columns] \
                + [i + 'diff9+24' for i in dataset.columns] \
                + [i + 'diff9+24*7' for i in dataset.columns] \
                + [i + 'diff24+24' for i in dataset.columns] \
                + [i + 'diff24+24*7' for i in dataset.columns] \
                + [i + '_mean_t-9-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-9-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-9-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-9-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-9-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-9-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns]
    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Temperature_Fujisan']

    return train_x, train_y


def get_data_hk_2h(dataset):

    df = pd.concat([pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1),
                    dataset.shift(2),
                    dataset.shift(3),
                    dataset.shift(4),
                    dataset.shift(5),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0).diff(),
                    dataset.shift(2).diff(),
                    dataset.shift(3).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=12).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=12).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=12).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day'] \
                + [i + '_t' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-2' for i in dataset.columns] \
                + [i + '_t-3' for i in dataset.columns] \
                + [i + '_t-4' for i in dataset.columns] \
                + [i + '_t-5' for i in dataset.columns] \
                + [i + 'diff0' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff2' for i in dataset.columns] \
                + [i + 'diff3' for i in dataset.columns] \
                + [i + '_mean_t-2-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-2-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-2-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \

    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Precipitation_Hakone']

    return train_x, train_y


def get_data_hk_7h(dataset):

    df = pd.concat([pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(5),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=12).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=12).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=12).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day'] \
                + [i + '_t' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-5' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff0' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_mean_t-7-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-7-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-7-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \

    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Precipitation_Hakone']

    return train_x, train_y


def get_data_hk_8h(dataset):

    df = pd.concat([pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(5),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=12).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=12).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=12).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day'] \
                + [i + '_t' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-5' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff0' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_mean_t-8-12_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-12_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-12_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \

    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Precipitation_Hakone']

    return train_x, train_y


def get_data_hk_9h(dataset):

    df = pd.concat([pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(5),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(0).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(1).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(2).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(3).diff(),
                    dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).shift(4).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=12).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=12).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=12).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day'] \
                + [i + '_t' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_t-5' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff0' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff1' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff2' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff3' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + 'diff4' for i in dataset.drop(columns=['Precipitation_Hakone', 'Temperature_Fujisan']).columns] \
                + [i + '_mean_t-8-12_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-12_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-12_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24*7_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24*7_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24*7_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \

    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Precipitation_Hakone']

    return train_x, train_y


def get_data_mtfuji_2h_nofuture(dataset):
    df = pd.concat([
                    pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.sin(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    dataset.shift(2),
                    dataset.shift(3),
                    dataset.shift(4),
                    dataset.shift(5),
                    dataset.shift(2).diff(),
                    dataset.shift(3).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day', 'sin_hour', 'cos_hour'] \
                + [i + '_t-2' for i in dataset.columns] \
                + [i + '_t-3' for i in dataset.columns] \
                + [i + '_t-4' for i in dataset.columns] \
                + [i + '_t-5' for i in dataset.columns] \
                + [i + 'diff2' for i in dataset.columns] \
                + [i + 'diff3' for i in dataset.columns] \
                + [i + '_mean_t-2-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-2-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns]
    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Temperature_Fujisan']

    return train_x, train_y


def get_data_mtfuji_7h_nofuture(dataset):
    df = pd.concat([
                    pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.sin(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    dataset.shift(7),
                    dataset.shift(8),
                    dataset.shift(9),
                    dataset.shift(23),
                    dataset.shift(24),
                    dataset.shift(25),
                    dataset.shift(7).diff(),
                    dataset.shift(8).diff(),
                    dataset.shift(24).diff(),
                    dataset.shift(7).diff(periods=24),
                    dataset.shift(7).diff(periods=24*7),
                    dataset.shift(24).diff(periods=24),
                    dataset.shift(24).diff(periods=24*7),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day', 'sin_hour', 'cos_hour'] \
                + [i + '_t-7' for i in dataset.columns] \
                + [i + '_t-8' for i in dataset.columns] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + '_t-23' for i in dataset.columns] \
                + [i + '_t-24' for i in dataset.columns] \
                + [i + '_t-25' for i in dataset.columns] \
                + [i + 'diff7' for i in dataset.columns] \
                + [i + 'diff8' for i in dataset.columns] \
                + [i + 'diff24' for i in dataset.columns] \
                + [i + 'diff7+24' for i in dataset.columns] \
                + [i + 'diff7+24*7' for i in dataset.columns] \
                + [i + 'diff24+24' for i in dataset.columns] \
                + [i + 'diff24+24*7' for i in dataset.columns] \
                + [i + '_mean_t-7-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-7-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns]
    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Temperature_Fujisan']

    return train_x, train_y


def get_data_mtfuji_8h_nofuture(dataset):
    df = pd.concat([
                    pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.sin(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    dataset.shift(8),
                    dataset.shift(9),
                    dataset.shift(10),
                    dataset.shift(23),
                    dataset.shift(24),
                    dataset.shift(25),
                    dataset.shift(8).diff(),
                    dataset.shift(9).diff(),
                    dataset.shift(24).diff(),
                    dataset.shift(8).diff(periods=24),
                    dataset.shift(8).diff(periods=24*7),
                    dataset.shift(24).diff(periods=24),
                    dataset.shift(24).diff(periods=24*7),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day', 'sin_hour', 'cos_hour'] \
                + [i + '_t-8' for i in dataset.columns] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + '_t-10' for i in dataset.columns] \
                + [i + '_t-23' for i in dataset.columns] \
                + [i + '_t-24' for i in dataset.columns] \
                + [i + '_t-25' for i in dataset.columns] \
                + [i + 'diff8' for i in dataset.columns] \
                + [i + 'diff9' for i in dataset.columns] \
                + [i + 'diff24' for i in dataset.columns] \
                + [i + 'diff8+24' for i in dataset.columns] \
                + [i + 'diff8+24*7' for i in dataset.columns] \
                + [i + 'diff24+24' for i in dataset.columns] \
                + [i + 'diff24+24*7' for i in dataset.columns] \
                + [i + '_mean_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns]
    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Temperature_Fujisan']

    return train_x, train_y


def get_data_mtfuji_9h_nofuture(dataset):
    df = pd.concat([
                    pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.sin(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.hour * 2 * np.pi/24), index=dataset.index),
                    dataset.shift(9),
                    dataset.shift(10),
                    dataset.shift(11),
                    dataset.shift(23),
                    dataset.shift(24),
                    dataset.shift(25),
                    dataset.shift(9).diff(),
                    dataset.shift(10).diff(),
                    dataset.shift(24).diff(),
                    dataset.shift(9).diff(periods=24),
                    dataset.shift(9).diff(periods=24*7),
                    dataset.shift(24).diff(periods=24),
                    dataset.shift(24).diff(periods=24*7),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day', 'sin_hour', 'cos_hour'] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + '_t-10' for i in dataset.columns] \
                + [i + '_t-11' for i in dataset.columns] \
                + [i + '_t-23' for i in dataset.columns] \
                + [i + '_t-24' for i in dataset.columns] \
                + [i + '_t-25' for i in dataset.columns] \
                + [i + 'diff9' for i in dataset.columns] \
                + [i + 'diff10' for i in dataset.columns] \
                + [i + 'diff24' for i in dataset.columns] \
                + [i + 'diff9+24' for i in dataset.columns] \
                + [i + 'diff9+24*7' for i in dataset.columns] \
                + [i + 'diff24+24' for i in dataset.columns] \
                + [i + 'diff24+24*7' for i in dataset.columns] \
                + [i + '_mean_t-9-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-9-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-9-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-9-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-9-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-9-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns]
    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Temperature_Fujisan']

    return train_x, train_y


def get_data_hk_2h_nofuture(dataset):

    df = pd.concat([pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    dataset.shift(2),
                    dataset.shift(3),
                    dataset.shift(4),
                    dataset.shift(5),
                    dataset.shift(2).diff(),
                    dataset.shift(3).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=12).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=12).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=12).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(2).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day'] \
                + [i + '_t-2' for i in dataset.columns] \
                + [i + '_t-3' for i in dataset.columns] \
                + [i + '_t-4' for i in dataset.columns] \
                + [i + '_t-5' for i in dataset.columns] \
                + [i + 'diff2' for i in dataset.columns] \
                + [i + 'diff3' for i in dataset.columns] \
                + [i + '_mean_t-2-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-2-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-2-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-2-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-2-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \

    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Precipitation_Hakone']

    return train_x, train_y


def get_data_hk_7h_nofuture(dataset):

    df = pd.concat([pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    dataset.shift(7),
                    dataset.shift(8),
                    dataset.shift(9),
                    dataset.shift(7).diff(),
                    dataset.shift(8).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=12).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=12).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=12).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(7).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day'] \
                + [i + '_t-7' for i in dataset.columns] \
                + [i + '_t-8' for i in dataset.columns] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + 'diff7' for i in dataset.columns] \
                + [i + 'diff8' for i in dataset.columns] \
                + [i + '_mean_t-7-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-12_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-7-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-24_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-7-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-7-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-7-24*7_t-7' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \

    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Precipitation_Hakone']

    return train_x, train_y


def get_data_hk_8h_nofuture(dataset):

    df = pd.concat([pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    dataset.shift(8),
                    dataset.shift(9),
                    dataset.shift(10),
                    dataset.shift(8).diff(),
                    dataset.shift(9).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=12).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=12).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=12).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(8).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day'] \
                + [i + '_t-8' for i in dataset.columns] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + '_t-10' for i in dataset.columns] \
                + [i + 'diff8' for i in dataset.columns] \
                + [i + 'diff9' for i in dataset.columns] \
                + [i + '_mean_t-8-12_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-12_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-12_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24*7_t-8' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \

    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Precipitation_Hakone']

    return train_x, train_y


def get_data_hk_9h_nofuture(dataset):

    df = pd.concat([pd.DataFrame(np.sin(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    pd.DataFrame(np.cos(dataset.index.day * 2 * np.pi/365), index=dataset.index),
                    dataset.shift(9),
                    dataset.shift(10),
                    dataset.shift(11),
                    dataset.shift(9).diff(),
                    dataset.shift(10).diff(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=12).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=12).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=12).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24).min(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).mean(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).max(),
                    dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].shift(9).rolling(window=24*7).min()],
                    axis=1)

    df.columns =  ['sin_day', 'cos_day'] \
                + [i + '_t-9' for i in dataset.columns] \
                + [i + '_t-10' for i in dataset.columns] \
                + [i + '_t-11' for i in dataset.columns] \
                + [i + 'diff9' for i in dataset.columns] \
                + [i + 'diff10' for i in dataset.columns] \
                + [i + '_mean_t-8-12_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-12_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-12_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_mean_t-8-24*7_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_max_t-8-24*7_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \
                + [i + '_min_t-8-24*7_t-9' for i in dataset[['Precipitation_Hakone', 'Temperature_Fujisan']].columns] \

    train_x = df.dropna()
    train_y = dataset.loc[train_x.index.values[0]:, 'Precipitation_Hakone']

    return train_x, train_y
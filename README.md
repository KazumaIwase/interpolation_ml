<div align="center">

# Interpolation of mountain weather forecasts by machine learning

Kazuma Iwase and Tomoyuki Takenawa

Graduate School of Marine Science and Technology, Tokyo University of
Marine Science and Technology, 2-1-6 Etchujima, Koto-ku, Tokyo,
135-8533, Japan.

Contributing authors: kazumaiwase676@gmail.com;
takenawa@kaiyodai.ac.jp;

<div align="left">

## About data

### Observed data
Observation data ([observe.csv](data/observe.csv)) is what we pre-processed from data downloaded from the JMA (Japan Meteorological Agency) website (https://www.data.jma.go.jp/risk/obsdl/index.php).

### Dummy data
There are csv files ([dummy_nesw.csv](data/dummy_nesw.csv), [dummy_tenki_to_kurasu.csv](data/dummy_tenki_to_kurasu.csv)) in the data folder, but they are dummy data.
Note that they are not the data used in this research.
Dummy data was created by adding an error to the observed data so that the RMSE after 8 hours match the actual forecast data.

## Usage
You can run some notebooks in [notebooks](notebooks) folder on googlecolab.

training_lightgbm.ipynb: LigtGBM training for forcasting Temperature at Mt.Fuji and Precipitation at Hakone. Forecast times can be selected from 2h, 7h, 8h and 9h ahead. 

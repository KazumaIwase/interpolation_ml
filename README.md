<div align="center">

# Interpolation of mountain weather forecasts by machine learning

Kazuma Iwase and Tomoyuki Takenawa

Faculty of Marine Technology, Tokyo University of Marine Science and
Technology, 2-1-6 Etchujima, Koto-ku, Tokyo, 135-8533, Japan.

<div align="left">

## About data

### Observed data
Observation data ([observe.csv](data/observe.csv)) were downloaded from the JMA(Japan Meteorological Agency) website (https://www.data.jma.go.jp/risk/obsdl/index.php). And it is what we pre-processed.

### Dummy data
There are csv files ([dummy_nesw.csv](data/dummy_nesw.csv), [dummy_tenki_to_kurasu.csv](data/dummy_tenki_to_kurasu.csv)) in the data folder, but they are dummy data.
Note that they are not the data used in this research.
Dummy data was created by adding an error to the observed data so that the RMSE after 8 hours match the actual forecast data.

## Usage
You can run some notebooks in [notebooks](notebooks) folder on googlecolab.

# Design Project

## Data Processing

### Aim

To create a tool which auto cleans and merges the datasets

### Datasets

- man_features (contains 12 csv files for each month)
- 2021_all_manchester_data_ADSB.csv
- all_met.csv
- flight_plan_manchester_2021.csv
- radar_manchester_2021.csv

### Files

- toolkit_dp.py: Does the datacleaning, dataprocessing and merging

## Modelling

### Aim

The aim of this repo is basically train DN modules so that everything fits properly.

### Datasets

- train.csv
- test.csv

### Progress Store

Done in dir - "./stores" and stores:

- graphs
- dnn models
- onnx of dnn modules
- scalars used for preprocessing

### Files

- modelss.ipynb: inorganised effort which suggests this is worthwhile to consider
- toolkit_dnn.py: Preprocessing, training, validation and ...
- dnn_v{n}.py: Tries
- test_model.py: testing model feature importance using PFI
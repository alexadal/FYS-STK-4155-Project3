# FYS-STK-4155-Project3
This is the repository for Project 3 in the course FYS-STK-4155, Fall 2019 at UiO.
The raw data analysed in the project is stored in the file rawdata_finance.xlsx
The code is structured in the following way:
1. To run parameter optimisation using Talos, run:
    - Talos_ANN.py for the Feed Forward Neural Network.
    - Talos_LSTM.py for the LSTM model.
2. To run paramater tuning for SVM, run tuning_svr.py
3. To recreate the plots from the paper, please run the file: create_price_plots.py
4. To print the best parameters for LSTM and FFNN, print_lstm.py and print_ffnn.py can be used respectively, by specifying the filename of the model to be loaded.
   The models are saved in the folder Talos_Results.

As the models are created using Tensorflow and SciKit-Learn, it has not been considered necessary to create unit tests for these at this point.

NOTE: In order to load saved models made using Talos on LSTM, the following changes have to be made in order to handle that LSTM uses 3-dimensional input.

In talos/utils/load_model.py:
![Image of first change](https://github.com/alexadal/FYS-STK-4155-Project3/blob/master/1.png)


In Talos/Commands/restore.py
![Image of second change](https://github.com/alexadal/FYS-STK-4155-Project3/blob/master/2.png)




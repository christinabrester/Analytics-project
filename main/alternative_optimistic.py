#---------------------------------------------------------------------------
#
# The main file to start the model training
# and testing in the Alternative and Optimistic scenarios.
# Results are saved in the folder 'results_observations'
#
#---------------------------------------------------------------------------

import os
import sys
import pandas as pd
import numpy as np
from preprocessing import preprocess
from experiment import ExperimentObservationsVsForecast
from neural_network import MLP, LSTMc, ArbitraryNN
from glob import glob

START_DATE = '2020-05-05 00:00'
END_DATE = '2020-11-05 00:00'

SELECTED_COLUMNS = ['UTC', 's_glob', 's_dif', 'T_AVG', 'wind_avg', 
                    'wind_dir_cos', 'wind_dir_sin', 'prec_amt', 
                    'cl_tot', 'power [W]', 'forecast fmi [W]']   

def main(argv): 
    # use command line arguments: df_file_obs, df_file_frcst, df_name
    df_obs = pd.read_excel(argv[0], index_col=0)
    df_frcst = pd.read_excel(argv[1], index_col=0)
    
    site_name = argv[2]
    
    df_obs = preprocess(df_obs, site_name, SELECTED_COLUMNS, START_DATE, END_DATE)
    df_frcst = preprocess(df_frcst, site_name, SELECTED_COLUMNS, START_DATE, END_DATE)
    
    # Test a bunch of MLP models
    data_obs = df_obs.copy()
    data_frcst = df_frcst.copy()
    fmi_input = False
    output_col = 'power [W]'
    models_saved = glob('results_observations/*') # 'results' folder contains results for all previously trained models
    
    for lookback in [1,2,3,6,9]:
        for nneurons in [ [4], [4,4], [8], [8,8], [16], [16,16], [32], [32,32] ]:
            for model_type in ['MLP' ]:
                experiment = ExperimentObservationsVsForecast( 
                                                                data_obs = data_obs,
                                                                data_frcst = data_frcst,
                                                                fmi_input = fmi_input,
                                                                lookback = lookback,
                                                                output_col = output_col
                                                                )


                
                # Define the model
                if model_type in ['MLP' , 'LSTM', 'ArbitraryNN']:
                    if model_type == 'MLP':
                        model = MLP(nneurons = nneurons)
                    elif model_type == 'LSTM':
                        model = LSTMc(nneurons = nneurons)
                    else:
                        model = ArbitraryNN()
                else:
                    raise ValueError("{} is not a valid type of the model. Choose 'MLP' , 'LSTM' or 'ArbitraryNN'".format(self.__model_type))

                # First, check if the result for the model is already in the folder
                nneurons_str = '.'.join(list(map(str, nneurons)))
                file_name1 = 'predictions_{}_{}obs_{}_{}_{}'.format(site_name, model_type, output_col, nneurons_str, lookback)
                file_name2 = 'validation_{}_{}obs_{}_{}_{}'.format(site_name, model_type, output_col, nneurons_str, lookback)

                if sum(list(map(lambda x: file_name1 in x or file_name2 in x, models_saved))):
                    print('Model trained: {}'.format('{}_{}obs_{}_{}_{}'.format(site_name, model_type, output_col, nneurons_str, lookback)))
                else:
                    print('Model not trained: {}'.format('{}_{}obs_{}_{}_{}'.format(site_name, model_type, output_col, nneurons_str, lookback)))
                    print('Run the experiment')
                    # Second, run the experiment if not
                    if not os.path.exists('results_observations'):
                        os.makedirs('results_observations')
                    experiment.run(model)
                    experiment.export_results(site_name, model_type, nneurons)


if __name__ == '__main__':
    main(sys.argv[1:])



















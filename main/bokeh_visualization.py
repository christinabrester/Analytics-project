#-----------------------------------------------------------------------------------
#
# Visualization of the results stored in the folder called 'results'
#
#-----------------------------------------------------------------------------------

from my_site import Site
from site_in_bokeh import build_dashboard
import pandas as pd
import numpy as np
from glob import glob
from bokeh.io import output_file, show
from preprocessing import preprocess

START_DATE = '2020-05-05 00:00'
END_DATE = '2020-11-05 00:00'


SELECTED_COLUMNS = ['UTC', 's_glob', 's_dif', 'T_AVG', 'wind_avg', 
                    'wind_dir_cos', 'wind_dir_sin', 'albedo', 'prec_amt', 
                    'cl_tot', 'cl_low', 'cl_med', 'cl_high',
                    'power [W]', 'forecast fmi [W]']

def get_best_model_inner_cv(site_name, cv_folds, models):
    '''
    Model selection in the inner loop of nested cross-validation based on 'val_loss'
    '''
    best_model_inner_cv = dict()
    for model in models:
        files = glob('results/validation_{}_{}_*'.format(site_name, model))
        inner_cv_model = pd.DataFrame(columns = ['run_test', 'val_loss', 'file_name'])
        for file in files:
            df = pd.read_csv(file, index_col=0)
            df = df.loc[:, ['run_test', 'val_loss']].groupby(['run_test']).mean().reset_index(drop = False)
            df['file_name'] = file
            
            inner_cv_model = pd.concat([inner_cv_model, df], ignore_index = True)

        dict_temp = dict()
        for run in range(cv_folds):
            min_loss = inner_cv_model.loc[inner_cv_model['run_test'] == run]['val_loss'].idxmin()
            dict_temp[run] = inner_cv_model.loc[min_loss, 'file_name']
            
        best_model_inner_cv[model] = dict_temp

    return best_model_inner_cv

    
def main(): 
    files = ['vuorela_solar_PV.xlsx', 'kuopio_solar_PV.xlsx', 'savilahti_solar_PV.xlsx']
    site_names = ['Vuorela', 'Kuopio', 'Savilahti']
    models = ['forecast fmi [W]', 'MLP', 'MLPfrcst', 'MLPobs']

    cv_folds = 5
    
    sites = []
    for i in range(len(site_names)):
        df = pd.read_excel(files[i], index_col=0)
        site_name = site_names[i]
        df = preprocess(df, site_name, ['UTC', 'forecast fmi [W]', 'power [W]'], START_DATE, END_DATE)

        best_model_inner_cv = get_best_model_inner_cv(site_name, cv_folds, models[1:])

        for model in models[1:]:
            df[model] = np.nan

            for run in range(cv_folds):
                file_name = 'results/predictions' + best_model_inner_cv[model][run].split('validation')[1]
                df_predictions = pd.read_csv(file_name, index_col = 0)

                indices = df_predictions.index[df_predictions['test_id'] == run]
                df.loc[indices, model] =  df_predictions.loc[df_predictions['test_id'] == run, 'prediction']

        df['test_id'] = df_predictions['test_id']
                           
        site = Site(
            name = site_name,
            dates=df['UTC'].values,
            power_collection = df,
            true_output_name = 'power [W]',
            models = models,
            test_id = df['test_id'].values
                   )

        sites.append(site)

        df.to_excel('best_predictions_{}.xlsx'.format(site_name))

    output_file("bokeh_visualization_obs_best_all.html", mode='inline')
    show(build_dashboard(sites))



if __name__ == '__main__':
    main()

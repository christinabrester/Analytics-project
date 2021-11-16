#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# A class Experiment implements the model training and testing on weather forecast data (the Baseline scenario)
# A class ExperimentObservationsVsForecast implements the model training on weather observations
# and two types of testing - on weather forecast data (the Alternative scenario) and on weather observations (the Optimistic scenario) 
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import my_tools
import pandas as pd
import numpy as np
import time
from metrics import evaluate_forecast, IA, RMSE, NRMSE1, NRMSE2, MAE


# Baseline scenario

class Experiment:
    BASIC_INPUTS = [
                                   's_glob', 's_dif', 'T_AVG', 'wind_avg', 
                                   'wind_dir_cos', 'wind_dir_sin', 'prec_amt', 'cl_tot',
                                   'day_cos', 'day_sin', 'hour_cos', 'hour_sin'
                                   ]
     
    NFOLDS_TEST = 5
    NFOLDS_VALID = 4
    PERIOD_TEST = 35 # running window in days 1..35 | 1..35 | 1..35 to split each interval into nfolds
    POWER_COL = 'power [W]'
    
    def __init__(self, **kwargs):
        self.input_cols = kwargs.get('input_cols', Experiment.BASIC_INPUTS[:])
        self.fmi_input = kwargs.get('fmi_input', False)
        if self.fmi_input == True:
            self.input_cols.append('forecast fmi [W]')
        self.output_col = kwargs.get('output_col', Experiment.POWER_COL)
        self.dates_col = kwargs.get('dates_col', 'UTC')
        self.fmi_forecast_col = kwargs.get('fmi_forecast_col', 'forecast fmi [W]')
        self.true_power_col = kwargs.get('power_col', 'power [W]')
        
        self.dataset = kwargs.get('data', pd.DataFrame())
        self.fmi_forecast = np.array([])
        if 'residual' in self.output_col:
            self.fmi_forecast = self.dataset[self.fmi_forecast_col].ravel()
        
        # keep only relevant columns 
        try:
            if self.true_power_col == self.output_col:
                self.dataset = self.dataset.loc[:, [self.dates_col] + self.input_cols + [self.output_col]]
            else:
                self.dataset = self.dataset.loc[:, [self.dates_col] + self.input_cols + [self.output_col] + [self.true_power_col]]
        except:
            raise ValueError("One of the necessary columns is not in the data: ", self.dates_col, self.input_cols, self.output_col)

        self.nfolds_test = Experiment.NFOLDS_TEST
        self.nfolds_valid = Experiment.NFOLDS_VALID
        self.period_test = Experiment.PERIOD_TEST
        
        self.lookback = kwargs.get('lookback', 1)
        self.test_id = []
        self.experiment_id = int(time.time())

        self.predictions = pd.DataFrame({'UTC': self.dataset[self.dates_col], 'prediction': [np.nan]*self.dataset.shape[0]})
        self.validation_loss = pd.DataFrame(columns = ['run_test', 'run_valid', 'val_loss', 'epochs'])
                         
    def generate_test_id(self):
        self.test_id = my_tools.get_cv_splits(self.dataset.shape[0], self.nfolds_test, self.period_test)

    def postprocess_predictions(self, predictions, fmi_forecast):
        if 'residual' in self.output_col:
            predictions = predictions + fmi_forecast
        return np.array([pred if pred >= 0 else 0 for pred in predictions])

    def update_results(self, prediction, run):
        if run == 0:
            prediction = np.append([np.nan]*(self.lookback-1), prediction)
        self.predictions.loc[(np.array(self.test_id) == run), 'prediction'] = prediction

    def export_results(self, site_name, model_type, nneurons):
        nneurons_str = '.'.join(list(map(str, nneurons)))
        file_name1 = 'results/predictions_{}_{}_{}_{}_{}_id{}.csv'.format(site_name, model_type, self.output_col, nneurons_str, self.lookback, self.experiment_id)
        self.predictions.to_csv(file_name1)
        file_name2 = 'results/validation_{}_{}_{}_{}_{}_id{}.csv'.format(site_name, model_type, self.output_col, nneurons_str, self.lookback, self.experiment_id)
        self.validation_loss.to_csv(file_name2)
            
    def run(self, NN):      
        self.generate_test_id()
        
        for run in range(self.nfolds_test):
            # Get training and test parts of the FMI forecast
            fmi_forecast_test = np.array([])
            if 'residual' in self.output_col:
                fmi_forecast_train, fmi_forecast_test = my_tools.get_train_and_test(self.fmi_forecast[(self.lookback-1):], self.test_id[(self.lookback-1):], run)

            # Scaling the data
            scaler_x, scaler_y = my_tools.fit_scalers(self.dataset, self.input_cols, self.output_col, self.test_id, run)
            dataset_scaled = my_tools.apply_scalers(self.dataset, self.input_cols, self.output_col, scaler_x, scaler_y)
            
            # Generate samples with previous observations
            samples_x, samples_y = my_tools.generate_samples_with_lookback(dataset_scaled, self.input_cols, self.output_col, self.lookback)

            # Separate training and test data for a particular run
            data_train_x, data_test_x = my_tools.get_train_and_test(samples_x, self.test_id[(self.lookback-1):], run)
            data_train_y, data_test_y = my_tools.get_train_and_test(samples_y, self.test_id[(self.lookback-1):], run)

            # Run k-fold validation to determine epochs and val_loss
            period = self.period_test - int(self.period_test/self.nfolds_test)
            cv_index = my_tools.get_cv_splits(data_train_x.shape[0], self.nfolds_valid, period)

            opt_epochs_list = []
            val_loss_list = []
            
            for cv_run in range(self.nfolds_valid):
                train_x, valid_x = my_tools.get_train_and_test(data_train_x, cv_index, cv_run)
                train_y, valid_y = my_tools.get_train_and_test(data_train_y, cv_index, cv_run)

                NN.build_model((self.lookback, len(self.input_cols)))
                (opt_epochs, val_loss) = NN.train_model_with_valid(train_x, train_y, valid_x, valid_y)               
                opt_epochs_list.append(opt_epochs)
                val_loss_list.append(val_loss)
                self.validation_loss = self.validation_loss.append({'run_test': run, 'run_valid': cv_run, 'val_loss': val_loss, 'epochs': opt_epochs}, ignore_index=True)

            opt_epochs = np.mean(opt_epochs_list)
            NN.build_model((self.lookback, len(self.input_cols)))
            NN.train_model(data_train_x, data_train_y, opt_epochs)

            predictions_test = my_tools.inverse_scale(NN.predict(data_test_x), scaler_y).ravel()
            predictions_test = self.postprocess_predictions(predictions_test, fmi_forecast_test)
            self.update_results(predictions_test, run)

        self.predictions['test_id'] = self.test_id
        print(self.predictions)



# Alternative and Optimistic scenarios

class ExperimentObservationsVsForecast:
    BASIC_INPUTS = [
                                   's_glob', 's_dif', 'T_AVG', 'wind_avg', 
                                   'wind_dir_cos', 'wind_dir_sin', 'prec_amt', 
                                   'cl_tot', 'day_cos', 'day_sin', 'hour_cos', 'hour_sin'
                                   ]
  
    NFOLDS_TEST = 5
    NFOLDS_VALID = 4
    PERIOD_TEST = 35 # running window in days 1..35 | 1..35 | 1..35 to split each interval into nfolds
    POWER_COL = 'power [W]'
    
    def __init__(self, **kwargs):
        self.input_cols = kwargs.get('input_cols', ExperimentObservationsVsForecast.BASIC_INPUTS[:])
        self.fmi_input = kwargs.get('fmi_input', False)
        if self.fmi_input == True:
            self.input_cols.append('forecast fmi [W]')
        self.output_col = kwargs.get('output_col', ExperimentObservationsVsForecast.POWER_COL)
        self.dates_col = kwargs.get('dates_col', 'UTC')
        self.fmi_forecast_col = kwargs.get('fmi_forecast_col', 'forecast fmi [W]')
        self.true_power_col = kwargs.get('power_col', 'power [W]')
        
        self.dataset_obs = kwargs.get('data_obs', pd.DataFrame())
        self.dataset_frcst = kwargs.get('data_frcst', pd.DataFrame())

        self.fmi_forecast = np.array([])
        if 'residual' in self.output_col:
            self.fmi_forecast = self.dataset_obs[self.fmi_forecast_col].ravel()
        
        # keep only relevant columns in self.dataset_obs
        try:
            if self.true_power_col == self.output_col:
                self.dataset_obs = self.dataset_obs.loc[:, [self.dates_col] + self.input_cols + [self.output_col]]
            else:
                self.dataset_obs = self.dataset_obs.loc[:, [self.dates_col] + self.input_cols + [self.output_col] + [self.true_power_col]]
        except:
            raise ValueError("One of the necessary columns is not in the data: ", self.dates_col, self.input_cols, self.output_col)

        # keep only relevant columns in self.dataset_frcst
        try:
            if self.true_power_col == self.output_col:
                self.dataset_frcst = self.dataset_frcst.loc[:, [self.dates_col] + self.input_cols + [self.output_col]]
            else:
                self.dataset_frcst = self.dataset_frcst.loc[:, [self.dates_col] + self.input_cols + [self.output_col] + [self.true_power_col]]
        except:
            raise ValueError("One of the necessary columns is not in the data: ", self.dates_col, self.input_cols, self.output_col)


        self.nfolds_test = ExperimentObservationsVsForecast.NFOLDS_TEST
        self.nfolds_valid = ExperimentObservationsVsForecast.NFOLDS_VALID
        self.period_test = ExperimentObservationsVsForecast.PERIOD_TEST
        
        self.lookback = kwargs.get('lookback', 1)
        self.test_id = []
        self.experiment_id = int(time.time())

        self.predictions_obs = pd.DataFrame({'UTC': self.dataset_obs[self.dates_col], 'prediction': [np.nan]*self.dataset_obs.shape[0]})
        self.predictions_frcst = pd.DataFrame({'UTC': self.dataset_frcst[self.dates_col], 'prediction': [np.nan]*self.dataset_frcst.shape[0]})
        self.validation_loss = pd.DataFrame(columns = ['run_test', 'run_valid', 'val_loss', 'epochs'])
                         
    def generate_test_id(self):
        self.test_id = my_tools.get_cv_splits(self.dataset_obs.shape[0], self.nfolds_test, self.period_test)

    def postprocess_predictions(self, predictions, fmi_forecast):
        if 'residual' in self.output_col:
            predictions = predictions + fmi_forecast
        return np.array([pred if pred >= 0 else 0 for pred in predictions])

    def update_results(self, prediction_obs, prediction_frcst, run):
        if run == 0:
            prediction_obs = np.append([np.nan]*(self.lookback-1), prediction_obs)
            prediction_frcst = np.append([np.nan]*(self.lookback-1), prediction_frcst)
        self.predictions_obs.loc[(np.array(self.test_id) == run), 'prediction'] = prediction_obs
        self.predictions_frcst.loc[(np.array(self.test_id) == run), 'prediction'] = prediction_frcst

    def export_results(self, site_name, model_type, nneurons):
        nneurons_str = '.'.join(list(map(str, nneurons)))
        # observations
        file_name1 = 'results_observations/predictions_{}_{}obs_{}_{}_{}_id{}.csv'.format(site_name, model_type, self.output_col, nneurons_str, self.lookback, self.experiment_id)
        self.predictions_obs.to_csv(file_name1)
        file_name2 = 'results_observations/validation_{}_{}obs_{}_{}_{}_id{}.csv'.format(site_name, model_type, self.output_col, nneurons_str, self.lookback, self.experiment_id)
        self.validation_loss.to_csv(file_name2)
        # forecast
        file_name1 = 'results_observations/predictions_{}_{}frcst_{}_{}_{}_id{}.csv'.format(site_name, model_type, self.output_col, nneurons_str, self.lookback, self.experiment_id+1)
        self.predictions_frcst.to_csv(file_name1)
        file_name2 = 'results_observations/validation_{}_{}frcst_{}_{}_{}_id{}.csv'.format(site_name, model_type, self.output_col, nneurons_str, self.lookback, self.experiment_id+1)
        self.validation_loss.to_csv(file_name2)
            
    def run(self, NN):      
        self.generate_test_id()
        
        for run in range(self.nfolds_test):
            # Get training and test parts of the FMI forecast
            fmi_forecast_test = np.array([])
            if 'residual' in self.output_col:
                fmi_forecast_train, fmi_forecast_test = my_tools.get_train_and_test(self.fmi_forecast[(self.lookback-1):], self.test_id[(self.lookback-1):], run)

            # Scaling the data
            scaler_x, scaler_y = my_tools.fit_scalers(self.dataset_obs, self.input_cols, self.output_col, self.test_id, run)
            dataset_scaled_obs = my_tools.apply_scalers(self.dataset_obs, self.input_cols, self.output_col, scaler_x, scaler_y)
            dataset_scaled_frcst = my_tools.apply_scalers(self.dataset_frcst, self.input_cols, self.output_col, scaler_x, scaler_y)
            
            # Generate samples with previous observations
            samples_x_obs, samples_y_obs = my_tools.generate_samples_with_lookback(dataset_scaled_obs, self.input_cols, self.output_col, self.lookback)
            samples_x_frcst, samples_y_frcst = my_tools.generate_samples_with_lookback(dataset_scaled_frcst, self.input_cols, self.output_col, self.lookback)

            # Separate training and test data for a particular run
            data_train_x_obs, data_test_x_obs = my_tools.get_train_and_test(samples_x_obs, self.test_id[(self.lookback-1):], run)
            data_train_y_obs, data_test_y_obs = my_tools.get_train_and_test(samples_y_obs, self.test_id[(self.lookback-1):], run)
            
            data_train_x_frcst, data_test_x_frcst = my_tools.get_train_and_test(samples_x_frcst, self.test_id[(self.lookback-1):], run)
            data_train_y_frcst, data_test_y_frcst = my_tools.get_train_and_test(samples_y_frcst, self.test_id[(self.lookback-1):], run)

            # Run k-fold validation to determine epochs and val_loss
            period = self.period_test - int(self.period_test/self.nfolds_test)
            cv_index = my_tools.get_cv_splits(data_train_x_obs.shape[0], self.nfolds_valid, period)

            opt_epochs_list = []
            val_loss_list = []
            
            for cv_run in range(self.nfolds_valid):
                train_x, valid_x = my_tools.get_train_and_test(data_train_x_obs, cv_index, cv_run)
                train_y, valid_y = my_tools.get_train_and_test(data_train_y_obs, cv_index, cv_run)

                NN.build_model((self.lookback, len(self.input_cols)))
                (opt_epochs, val_loss) = NN.train_model_with_valid(train_x, train_y, valid_x, valid_y)               
                opt_epochs_list.append(opt_epochs)
                val_loss_list.append(val_loss)
                self.validation_loss = self.validation_loss.append({'run_test': run, 'run_valid': cv_run, 'val_loss': val_loss, 'epochs': opt_epochs}, ignore_index=True)

            opt_epochs = np.mean(opt_epochs_list)
            NN.build_model((self.lookback, len(self.input_cols)))
            NN.train_model(data_train_x_obs, data_train_y_obs, opt_epochs)

            predictions_test_obs = my_tools.inverse_scale(NN.predict(data_test_x_obs), scaler_y).ravel()
            predictions_test_obs = self.postprocess_predictions(predictions_test_obs, fmi_forecast_test)

            predictions_test_frcst = my_tools.inverse_scale(NN.predict(data_test_x_frcst), scaler_y).ravel()
            predictions_test_frcst = self.postprocess_predictions(predictions_test_frcst, fmi_forecast_test)
            
            self.update_results(predictions_test_obs, predictions_test_frcst, run)

        self.predictions_obs['test_id'] = self.test_id
        self.predictions_frcst['test_id'] = self.test_id
        print(self.predictions_obs)



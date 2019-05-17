import numpy as np
import pandas as pd 
import logging
class OutlierTreatment:
    """

    Cap Outlier Data for continuous variables.

    Parameters
    ----------
    attr: Dictionary containing following values.

       - dset: dataset name( dev or itv or otv or test)
       - version: version number to save the output
       - dtype_loc: file path containing data type 

    """

    def __init__(self):
        logging.debug("inside Outlier treatment constructor")
        self.version = None
    '''
    def fit(self, df, basic_dict, lower_cap = 1, upper_cap = 99, version=None):
        logging.debug("inside fit function of outlier treatment class.  lower_cap: {}, upper_cap: {}".format(lower_cap, upper_cap))
        """
        Creates a fit object on the suppied df
        
        Parameter:
        ----------
        df : Input Dataframe to learn dummy variables (should be the development dataframe) 
        lower_cap : The lower bound percentile value to cap values. Default : 1  (1%)
        upper_cap : The lower bound percentile value to cap values. Default : 99 (99%)
        
        Values of lower and upper cap should be between 0 and 100.
        """

        self.version=version
        dtypes_num=basic_dict['num']

        #create a summary table for outlier treatment 
        outlier_treatment_df = pd.DataFrame(columns=['var_name', 'Lower_Cap', 'Upper_Cap', 'Lower Bound', 'Upper Bound', 
                                                    'Min', 'P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99', 'Max'])

        for var in dtypes_num:
            #_out_cap = out_treat_df[out_treat_df.Variable == var]["outlier_cap"].values[0]
            _lower_cap = lower_cap
            _upper_cap = upper_cap
            temp_df = pd.DataFrame(data={'var_name' : var, 
                                         'Lower_Cap' : _lower_cap,
                                         'Upper_Cap' : _upper_cap,
                                         'Lower Bound' : df[var].quantile(_lower_cap/100.0), 
                                         'Upper Bound' : df[var].quantile(_upper_cap/100.0), 
                                         'Min' : df[var].min(),
                                         'P1' : df[var].quantile(1.0/100),
                                         'P5' : df[var].quantile(5.0/100),
                                         'P10' : df[var].quantile(10.0/100),
                                         'P25' : df[var].quantile(25.0/100),
                                         'P50' : df[var].quantile(50.0/100),
                                         'P75' : df[var].quantile(75.0/100), 
                                         'P90' : df[var].quantile(90.0/100),
                                         'P95' : df[var].quantile(95.0/100),
                                         'P99' : df[var].quantile(99.0/100),
                                         'Max' : df[var].max()}, 
                                   columns=['var_name', 'outlier_cap', 'Max', 'Upper Bound', 'Lower Bound', 'Min'], 
                                   index=[var])

            outlier_treatment_df = outlier_treatment_df.append(temp_df)

        outlier_treatment_df.to_csv('IOFiles/outlier_treatment_df.csv', index=False)
        proceed = 'n'
        while proceed.lower() != 'y':
            proceed = input('\noutlier_treatment.csv file containing outlier treatment details is created. Make necessary changes and confirm to proceed (Y/N) -------->    ')

        outlier_treatment_df = pd.read_csv('IOFiles/outlier_treatment_df.csv')
        
        #modify summary table for outlier treatment 
        
        for var in dtypes_num:
            
            _lower_cap = outlier_treatment_df[outlier_treatment_df.Variable == var]["Lower_Cap"].values[0]
            _upper_cap = outlier_treatment_df[outlier_treatment_df.Variable == var]["Upper_Cap"].values[0]
            
            outlier_treatment_df.loc[outlier_treatment_df.Variable == var, 'Lower Bound'] = df[var].quantile(_lower_cap/100.0)
            outlier_treatment_df.loc[outlier_treatment_df.Variable == var, 'Upper Bound'] = df[var].quantile(_upper_cap/100.0)
            

        outlier_treatment_df.to_csv('IOFiles/outlier_treatment_df.csv', index=False)
        logging.debug("outlier treatment executed successfuly. dictionary is :{}".format(basic_dict))
    '''
    
    def fit(self, df, basic_dict, outlier_cap = 0.99, version=None):
        logging.debug("inside fit function of outlier treatment class.  outlier_cap: {}".format(outlier_cap))
        """
        Creates a fit object on the suppied df
        
        Parameter:
        ----------
        df : Input Dataframe to learn dummy variables (should be the development dataframe) 
        outlier_cap : The percentile value to cap values. Should be (ideally) between 0.9 and 1. 
                      For no outlier capping set value to 1. Default : 0.99
        """

        self.version=version
        dtypes_num=basic_dict['num']

        out_treat_df = pd.DataFrame({'Variable': dtypes_num})
        out_treat_df['outlier_cap'] = outlier_cap
        out_treat_df.to_csv('IOFiles/out_treat_df.csv', index=False)
        proceed = 'n'
        while proceed.lower() != 'y':
            proceed = input('\noutlier_treatment.csv file containing outlier treatment details is created. Make necessary changes and confirm to proceed (Y/N) -------->    ')

        out_treat_df = pd.read_csv('IOFiles/out_treat_df.csv')
		
        #create a summary table for outlier treatment 
        outlier_treatment_df = pd.DataFrame(columns=['var_name', 'outlier_cap', 'Max', 'Upper Bound', 'Lower Bound', 'Min'])

        for var in dtypes_num:
            _out_cap = out_treat_df[out_treat_df.Variable == var]["outlier_cap"].values[0]
            temp_df = pd.DataFrame(data={'var_name' : var, 
                                         'outlier_cap' : _out_cap,
                                         'Max' : df[var].max(), 
                                         'Upper Bound' : df[var].quantile(_out_cap), 
                                         'Lower Bound' : df[var].quantile(1 - _out_cap), 
                                         'Min' : df[var].min()}, 
                                   columns=['var_name', 'outlier_cap', 'Max', 'Upper Bound', 'Lower Bound', 'Min'], 
                                   index=[var])

            outlier_treatment_df = outlier_treatment_df.append(temp_df)

        outlier_treatment_df.to_csv('IOFiles/outlier_treatment_df.csv', index=False)
        logging.debug("outlier treatment executed successfuly. dictionary is :{}".format(basic_dict))

    def transform(self, df):
        logging.debug("inside transform module of Outlier treatment class.")
        """
        Treat numeric variables as per previous fit file
        
        Parameter:
        ----------
        df : Input Dataframe to create dummies 

        """
        outlier_treatment_df = pd.read_csv('IOFiles/outlier_treatment_df.csv', index_col='var_name')

        for var in outlier_treatment_df.index:
            _upper_bound = outlier_treatment_df[outlier_treatment_df.index == var]["Upper Bound"].values[0]
            _lower_bound = outlier_treatment_df[outlier_treatment_df.index == var]["Lower Bound"].values[0]

            df.loc[df[var] > _upper_bound, var] = _upper_bound
            df.loc[df[var] < _lower_bound, var] = _lower_bound

        return df

"""
Version v0.9.2

"""
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
        self.outlier_treatment=None
        self.basic_dict=basic_dict
        dtypes_num=basic_dict['num'] 
        outlier_treatment_df = pd.DataFrame(columns=['var_name', 'Lower_Cap', 'Upper_Cap', 'Lower Bound', 'Upper Bound','Min', 'P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99', 'Max'])
        for var in dtypes_num:
            _lower_cap = lower_cap
            _upper_cap = upper_cap
            temp_df = pd.DataFrame(data={'var_name' : var, 'Lower_Cap' : _lower_cap,'Upper_Cap' : _upper_cap,'Lower_Bound' : df[var].quantile(float(_lower_cap)/100.0), 'Upper_Bound' : df[var].quantile(float(_upper_cap)/100.0), 'Min' : df[var].min(),'P1' : df[var].quantile(1.0/100),'P5' : df[var].quantile(5.0/100),'P10' : df[var].quantile(10.0/100),'P25' : df[var].quantile(25.0/100),'P50' : df[var].quantile(50.0/100),'P75' : df[var].quantile(75.0/100), 'P90' : df[var].quantile(90.0/100),'P95' : df[var].quantile(95.0/100),'P99' : df[var].quantile(99.0/100),'Max' : df[var].max()}, columns=['var_name','Lower_Cap','Upper_Cap','Lower_Bound','Upper_Bound','Min','Max','P1','P5','P10','P25','P50','P75','P90','P95','P99'], index=[var])

            outlier_treatment_df = outlier_treatment_df.append(temp_df)

        outlier_treatment_df.to_csv(basic_dict['path']+'/'+'outlier_treatment_df_pre.csv',columns=['var_name','Lower_Cap','Upper_Cap','Lower_Bound','Upper_Bound','Min','Max','P1','P5','P10','P25','P50','P75','P90','P95','P99'], index=False)
        proceed = 'n'
        while proceed.lower() != 'y':
            proceed = input('Please make changes to outlier_treatment_df_pre.csv and confirm to proceed (Y/N):')
        outlier_treatment_df = pd.read_csv(basic_dict['path']+'/'+'outlier_treatment_df_pre.csv')
        #modify summary table for outlier treatment 
        for var in dtypes_num:
            _lower_cap = outlier_treatment_df[outlier_treatment_df.var_name == var]["Lower_Cap"].values[0]
            _upper_cap = outlier_treatment_df[outlier_treatment_df.var_name == var]["Upper_Cap"].values[0]
            outlier_treatment_df.loc[outlier_treatment_df.var_name == var, 'Lower_Bound'] = df[var].quantile(float(_lower_cap)/100.0)
            outlier_treatment_df.loc[outlier_treatment_df.var_name == var, 'Upper_Bound'] = df[var].quantile(float(_upper_cap)/100.0)
        self.outlier_treatment=outlier_treatment_df 
        outlier_treatment_df.to_csv(basic_dict['path']+'/'+'outlier_treatment_df_post.csv',columns=['var_name','Lower_Cap','Upper_Cap','Lower_Bound','Upper_Bound','Min','Max','P1','P5','P10','P25','P50','P75','P90','P95','P99'], index=False)
        logging.debug("outlier treatment executed successfuly. dictionary is :{}".format(basic_dict))

    def transform(self, df):
        logging.debug("inside transform module of Outlier treatment class.")
        """
        Treat numeric variables as per previous fit file
        
        Parameter:
        ----------
        df : Input Dataframe to create dummies 

        """
        outlier_treatment_df = pd.read_csv(self.basic_dict['path']+'/'+'outlier_treatment_df_post.csv', index_col='var_name')
        for var in outlier_treatment_df.index:
            _upper_bound = outlier_treatment_df[outlier_treatment_df.index == var]["Upper_Bound"].values[0]
            _lower_bound = outlier_treatment_df[outlier_treatment_df.index == var]["Lower_Bound"].values[0]
            df.loc[df[var] > _upper_bound, var] = _upper_bound
            df.loc[df[var] < _lower_bound, var] = _lower_bound
        return df


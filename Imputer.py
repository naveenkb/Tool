import pandas as pd
import numpy as np
import logging
class Imputer():
    """ Missing values Imputation
    
    Parameters:
    -----------
    attr: Dictionary containing following values.
            
       - dset: dataset name( dev or itv or otv or test)
       - version: version number to save the output
       - dtype_loc: file path containing data type 
       - impute_vals: Used to store imputation values after fit
    
    """
    def __init__(self):
        #self.dset = attr['dset']
        self.version = None
        #self.dtype_loc = attr['dtype_loc']
        self.impute_vals = None 
        logging.debug("Object of Imputer class Created")
        
    def fit(self, df,basic_dict,version=None):
        """
        Fits the imputer on df
        
        Parameter:
        ----------
        df : Input Dataframe to learn missing imputation 
        """
        logging.debug("inside fit function of Imputer Class for version{}".format(version))
        self.version=version
        #dtypes = pd.read_csv(self.dtype_loc)
        dtypes_num=pd.DataFrame({"col":basic_dict['num'],"dtype":"num"})
        dtypes_cat=pd.DataFrame({"col":basic_dict['cat'],"dtype":"cat"})
        dtypes=pd.concat([dtypes_num,dtypes_cat],ignore_index=True)
        dtypes['default_impute_type'] = np.where(dtypes.dtype == 'num', 'median', np.where(dtypes.dtype == 'cat', 'mode', np.nan))
        dtypes['user_impute_type'] = np.where(dtypes.dtype == 'num', 'median', np.where(dtypes.dtype == 'cat', 'mode', np.nan))
        dtypes = self._custom_impute(df, dtypes, 'default')
        dtypes.to_csv('IOFiles/impute_{}.csv'.format(self.version), index =False)
        proceed = 'n'
        while proceed.lower() != 'y':
            proceed = input('\nimpute_{}.csv file containing missing value imputation details has been created. Make necessary changes and confirm to proceed (Y/N) -------->    '.format(self.version))
        dtypes = self._custom_impute(df, dtypes, 'user')
        dtypes.to_csv('IOFiles/impute_{}.csv'.format(self.version), index =False)
        
        logging.debug("impute vals defined from devlopment data sets are {}".format(dtypes))
        self.impute_vals =  dtypes.user_impute_value
        self.impute_vals.index = dtypes.col
        logging.debug("Fitting on Development data Finished. Dictionary is : {} ".format(basic_dict))
            
    def transform(self, df):
        """
        Imputes missing values in given dataframe
        
        Parameter:
        ----------
        df : Input Dataframe to apply imputation 
        """
        logging.debug("inside transform function of Imputer. Values for Imputation are : {}".format(self.impute_vals) )
        if self.impute_vals is None:
            raise("Not Fitted Error")
        df = df.fillna(self.impute_vals)
        return df
        
    def _ret_zero(a):
        logging.debug("inside _ret_zero Module of Imputer class")
        return len(a) - len(a)
    
    def _custom_impute(self, df, dtypes, impute_id):
        """
        Depending on the type of column and correposing imputation type, imputation value is calculated and returned in the dataframe 
        
        Parameter:
        ----------
        df : Input Dataframe on which imputatuon value needs to be learned 
        dtypes : dataframe with user/default imputation type
        impute_id : user defined or default imputation to use 
        
        """
        logging.debug("inside _custome_imputer Module of Imputer Class")
        dtype2 = dtypes[dtypes.dtype.isin(['num', 'cat'])].reset_index(drop = True)
        if dtype2[impute_id + '_impute_type'].isnull().sum() > 0:
            print(impute_id + '_impute_type contains missing values')
        impute_dict = {dtype2['col'][i]: dtype2[impute_id + '_impute_type'][i] for i in range(dtype2.shape[0])}
        #impute_values = df.agg(impute_dict);
        
        #if isinstance(impute_values, pd.DataFrame):
            #impute_values = impute_values.iloc[0,:]
        impute_values={}
        for key,val in impute_dict.items():
            impute_values[key]=self.aggregate(df,key,val)
        dtypes[impute_id + '_impute_value'] = dtypes.col.map(impute_values)
        return dtypes
    
    def aggregate(self,df,key,val):
        logging.debug("inside aggregate Module of Imputer.")
        if val=="mean":
            return df[key].mean()
        elif val=="median":
            return df[key].median()
        elif val=="mode":
            return df[key].mode()[0]

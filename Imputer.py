"""
Version v0.9.2

"""
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
        self.dtypes=None
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
        self.dtypes=pd.concat([dtypes_num,dtypes_cat],ignore_index=True)
        self.dtypes['default_impute_type'] = np.where(self.dtypes.dtype == 'num', 'median', np.where(self.dtypes.dtype == 'cat', 'mode', np.nan))
        self.dtypes['user_impute_type'] = np.where(self.dtypes.dtype == 'num', 'median', np.where(self.dtypes.dtype == 'cat', 'mode', np.nan))
        NMiss=df[self.dtypes['col'].tolist()].isnull().sum(axis = 0).to_dict()
        self.dtypes['N_Miss'] = self.dtypes.col.map(NMiss)
        Ntotal=df.shape[0]
        self.dtypes['N_Miss_Perc'] = self.dtypes['N_Miss']*100/Ntotal
        self.dtypes = self._custom_impute(df,'default')
        self.dtypes['user' + '_impute_value'] = self.dtypes['default' + '_impute_value']
        self.dtypes.to_csv(basic_dict['path']+'/'+'impute_pre_{}.csv'.format(self.version), index =False)
        proceed = 'n'
        while proceed.lower() != 'y':
            proceed = input('\n impute_pre_{}.csv file containing missing value imputation details has been created. Make necessary changes and confirm to proceed (Y/N) -------->    '.format(self.version))
        self.dtypes =pd.read_csv(basic_dict['path']+'/'+'impute_pre_{}.csv'.format(self.version))
        self.dtypes = self._custom_impute(df,'user')
        self.dtypes=self.dtypes.drop(['N_Miss','N_Miss_Perc'],axis=1)
        self.dtypes.to_csv(basic_dict['path']+'/'+'impute_post_{}.csv'.format(self.version), index =False)
        print('description of Imputes post missing treatment is placed as impute_post_v.csv')
        logging.debug("impute vals defined from devlopment data sets are {}".format(self.dtypes))
        self.impute_vals =  self.dtypes.user_impute_value
        self.impute_vals.index = self.dtypes.col
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
    
    def _custom_impute(self, df,impute_id):
        """
        Depending on the type of column and correposing imputation type, imputation value is calculated and returned in the dataframe 
        
        Parameter:
        ----------
        df : Input Dataframe on which imputatuon value needs to be learned 
        dtypes : dataframe with user/default imputation type
        impute_id : user defined or default imputation to use 
        
        """
        logging.debug("inside _custome_imputer Module of Imputer Class")
        dtype2 = self.dtypes[self.dtypes.dtype.isin(['num', 'cat'])].reset_index(drop = True)
        if dtype2[impute_id + '_impute_type'].isnull().sum() > 0:
            print(impute_id + '_impute_type contains missing values')
        impute_dict = {dtype2['col'][i]: dtype2[impute_id + '_impute_type'][i] for i in range(dtype2.shape[0])}
        #impute_values = df.agg(impute_dict);
        
        #if isinstance(impute_values, pd.DataFrame):
            #impute_values = impute_values.iloc[0,:]
        impute_values={}
        for key,val in impute_dict.items():
            res=self.aggregate(df, key, val)
            if res=="nexist":
                pass
            else:
                impute_values[key] = res
        self.dtypes[impute_id + '_impute_value'] = self.dtypes.col.map(impute_values)
        return self.dtypes
    
    def aggregate(self,df,key,val):
        logging.debug("inside aggregate Module of Imputer.")
        if val=="mean":
            return df[key].mean()
        elif val=="median":
            return df[key].median()
        elif val=="mode":
            if len(df[key].mode())==0:
                print('the varible {0} can not be imputed by mode because all values are distinct and mode does not exist'.format(key))
                return 'nexist'
            else:
                return df[key].mode()[0]
        elif val=='zero':
            if self.dtypes.loc[self.dtypes['col'] == key, 'dtype'].iloc[0] =='num' :
                return 0
            if self.dtypes.loc[self.dtypes['col'] == key, 'dtype'].iloc[0] =='cat' :
                return str(0)
        elif val=='constant':
            if self.dtypes.loc[self.dtypes['col'] == key, 'dtype'].iloc[0] =='num' :
                return float(self.dtypes.loc[self.dtypes['col'] == key, 'user_impute_value'].iloc[0])
            if self.dtypes.loc[self.dtypes['col'] == key, 'dtype'].iloc[0] =='cat' :
                return str(self.dtypes.loc[self.dtypes['col'] == key, 'user_impute_value'].iloc[0])
        

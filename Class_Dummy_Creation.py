import numpy as np
import pandas as pd
import logging
class CreateDummies():
    """ 
    
    Create Dummy Variables for Categorical Data
    
    Parameters:
    -----------
    attr: Dictionary containing following values.
            
       - dset: dataset name( dev or itv or otv or test)
       - version: version number to save the output
       - dtype_loc: file path containing data type 
       - treat_vals: Used to store dummy values after fit
       - max_cat_levels : Maximum levels for the categorical variable to create dummies
       - na_dummies : Boolean value determining whether to create a new level for NAs
    
    """
    def __init__(self):
        logging.debug("inside constructor of Dummy Creation class Module")
        self.version = None
        self.na_dummies = None 
        
    def fit(self, df, basic_dict, max_cat_levels = 50, na_dummies = True, version=None):
        logging.debug("inside fit  Module of Dummy Creation class")
        """
        Creates a fit object on the suppied df
        
        Parameter:
        ----------
        df : Input Dataframe to learn dummy variables (should be the development dataframe) 
        max_cat_levels : Maximum levels for the categorical variable to create dummies
        na_dummies : Boolean value determining whether to create a new level for NAs
        """
        self.version=version
        
        #dtypes_num=pd.DataFrame({"col":basic_dict['num'],"dtype":"num"})
        dtypes_cat=basic_dict['cat']
        dummy_treatment_df = pd.DataFrame(columns=['var_name', 'no_of_levels', 'unique_levels'])
        for var in dtypes_cat:
            temp_df = pd.DataFrame(data={'var_name' : var, 
                                        'no_of_levels' : df[var].nunique(),
                                        'unique_levels' : [df[var].unique()]
                                        }, 
                                    columns=['var_name', 'no_of_levels', 'unique_levels'], 
                                    index=[var])
            dummy_treatment_df = dummy_treatment_df.append(temp_df)

        dummy_treatment_df['dummies_created'] = np.where(dummy_treatment_df['no_of_levels'] <= max_cat_levels, 'Yes', 'No')
        dummy_treatment_df['na_dummies'] = na_dummies

        # save the file
        dummy_treatment_df.to_csv('IOFiles/dummy_treatment_df.csv', index=False)

        self.na_dummies =  na_dummies
        logging.debug("dummies created are : {}".format(dummy_treatment_df['dummies_created']))
        #self.impute_vals.index = dtypes.col
            
    def transform(self, df):
        logging.debug("inside transform module of dummy Creation class.")
        """
        Create dummy variables as per previous fit file
        
        Parameter:
        ----------
        df : Input Dataframe to create dummies 
    
        """
        dummy_treatment_df = pd.read_csv('IOFiles/dummy_treatment_df.csv')
        for var in dummy_treatment_df.var_name:
            # create the dummies and append to dataset 
            if (dummy_treatment_df[dummy_treatment_df['var_name'] == var]['dummies_created'].values == 'Yes'):
                temp=pd.get_dummies(df[var], prefix=var, dummy_na=self.na_dummies, columns=var)
                df = pd.concat([df, temp], axis=1)

            # drop the categorical vars   
            df.drop(var, axis=1, inplace=True)
        
        return df 

"""
Version v0.9.2

"""
import logging
logging.captureWarnings(True)
import numpy as np
import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals import joblib
from xgboost import XGBClassifier
from itertools import chain, product
import pandas.core.algorithms as algos
import scipy.stats.stats as stats
import statsmodels.api as sm
import collections
import matplotlib as matplotlib
matplotlib.use("agg") 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pandas import ExcelWriter

class XgbSelection:
    """
    Parameters  : 
        
    """
    def __init__(self,description_dictionary,version):
        print("Constructor of XgbSelection has been called")
        #logging.debug("XgbSelection Class Object created")
        self.df=None
        self.dictionary=description_dictionary
        self.version=str(version)
        matplotlib.use('agg')
        matplotlib.rcParams.update({'font.size': 7})
        rcParams.update({'figure.autolayout': True})
        plt.style.use('seaborn-paper')
        if not os.path.exists(self.dictionary['path']+'/'+'results'):
            os.makedirs(self.dictionary['path']+'/'+'results')
 
        if not os.path.exists(self.dictionary['path']+'/'+'saved_objects'):
            os.makedirs(self.dictionary['path']+'/'+'saved_objects')
 
        if not os.path.exists(self.dictionary['path']+'/'+'PDP'):
            os.makedirs(self.dictionary['path']+'/'+'PDP')    
            
    def runInitializer(self):
        logging.debug("inside runInitializer function of XgbSelection class.")
        self.put_ParamSpace()
        print("XgbParamSpace.csv file has been placed at path :  " ,self.dictionary['path'] ,' Please provide required parameter space.' )
        action = input("\nConfirm if XGBoost parameter details are provided in XgbParamSpace.csv file and saved in the same location (Y/N) -------->    ").lower()
        while(action=='n'):
            action=input("Confirm if XGBoost parameter details are provided in XgbParamSpace.csv file and saved in the same location (Y/N) -------->    ").lower()
        self.get_ParamSpace()
        print("The parameter space for the grid search has been read")        
    """ put_ParamSpace is used to provide a file to modeller to specify the parameter search space :
          
    """
    def put_ParamSpace(self):
        logging.debug("inside put_Paramspace module of XgbSelection class. dictionary is :{} ".format(self.dictionary))
        default_param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [4, 6],
                        'subsample': [0.9, 1.0],
                        'learning_rate': [0.01, 0.1],
                        'colsample_bytree': [ 0.9, 1.0],
                        'gamma': [0.0,10.0],
                        'min_child_weight': [1,2]                    
                    }
        default_params_df = pd.DataFrame(default_param_grid, columns=default_param_grid.keys())
        default_params_df.to_csv(self.dictionary['path']+'/'+'XgbParamSpace.csv',index=False)
        logging.debug("Default params for XgbSelection class are:{} ".format(default_params_df))

        '''
        Read the Basic Requirement file and create Dynamic variables for all the basic fileds which are available in basic_requirements file.
        '''        
    def get_ParamSpace(self):
        logging.debug("inside get_ParamSpace Module of XgbSelection Class. dictionary is :{} ".format(self.dictionary))
        params_df=pd.read_csv(self.dictionary['path']+'/'+'XgbParamSpace.csv')
        params_df = params_df.where((pd.notnull(params_df)), None)
        parameters=params_df.columns.tolist()
        self.dictionary['XgbParam_grid']={parameters[par]:list(params_df.loc[:,parameters[par]]) for par in range(len(parameters))}
        self.check_Path()
        self.params_df = pd.DataFrame(list(product(*self.dictionary['XgbParam_grid'].values())), columns=self.dictionary['XgbParam_grid'].keys())
        self.params_df = self.params_df.loc[:,['n_estimators', 'max_depth', 'subsample', 'learning_rate', 'colsample_bytree','gamma','min_child_weight']]
        self.params_df.dropna(inplace=True)
        self.params_df.reset_index(drop=True, inplace=True)
        for col in self.params_df.columns:
            if (self.params_df[col].astype(float) == self.params_df[col].astype(float).astype(int)).all():
                self.params_df[col] = self.params_df[col].astype(int)
            else:
                self.params_df[col] = self.params_df[col].astype(float)
        logging.debug("Params for XgbSelection after user input are: {}: ".format(params_df))
       
    '''
    Check for the Path accesses, User must have read and write access for the Directory. 
    if User Dont have access for the Directory then we will not be able to process Further
    '''
    def check_Path(self,path_new=None):
        logging.debug("inside check_path module of XgbSelection Class . dictionary is : {}".format(self.dictionary))
        if(path_new!=None):
            self.dictionary['path']=path_new
            
        self.dictionary['path']='/'.join(self.dictionary['path'].split('/'))
        #self.path="/"+self.path
        try:
            read_f=0
            write_f=0
            if(os.access(self.dictionary['path'], os.R_OK)):
                read_f=1
            if(os.access(self.dictionary['path'], os.W_OK)):
                write_f=1
            if(not read_f and not write_f):
                print("Directory does not exist or You dont have (Read and write) access for the given ",self.dictionary['path'])
                return False
        except:
            print("Check path access")
            return False
    
    """ Performs grid search of Xgb parameters,for given feature combination 
    and saves the corresponding model objects and performance metrics['dev_ks', 
    'dev_ro_break', 'dev_ks_decile', 'dev_capture'] on dev-data
    Parameters:
    ----------
    df : Input dataframe(dev data)
    imp_features : list of important features to build the model
    """        
    def saveTrainingsDev(self,df,imp_features,save_pred = False):
        logging.debug("inside saveTrainingsDev Module of XgbSelection Class . dictionary is :  {}".format(self.dictionary))
        print('Iterating on different hyper parameters..')
        version=self.version
        out = df.loc[:, self.dictionary['id'] + self.dictionary['performance']]
        out['actual'] = df[self.dictionary['target'][0]]
        summary_df = pd.DataFrame()
        identifier = str(len(imp_features)) + 'var'
        alias = {
                'n_estimators': 'est',
                'max_depth': 'max_dep',
                'subsample': 'sub_s',
                'learning_rate': 'learn_r',
                'colsample_bytree': 'col_samp',
    	          'reg_lambda': 'lambda',
                  'gamma':'gamma',
                  'min_child_weight':'mcw'
    	           }
        for idx, row in self.params_df.astype(object).iterrows():
            print('Iteration {0} of {1}'.format(idx +1, self.params_df.shape[0]))
            tup = [i for i in zip([alias.get(row.index[j]) for j in range(len(self.params_df.columns))], row.values.astype(str))]
            params_str = [''.join(t) for t in tup]
            identifier = identifier  +'_'.join(params_str) + '_' + version
            param = row.to_dict()
            #model = XGBClassifier(seed = 10, **params, nthread = 10)
            model = XGBClassifier(seed = 10,learning_rate=param['learning_rate'],colsample_bytree=param['colsample_bytree'],
                                  n_estimators=param['n_estimators'],subsample=param['subsample'],max_depth=param['max_depth'],
                                                    gamma=param['gamma'] ,min_child_weight=param['min_child_weight'],nthread = 10)
            model.fit(df.loc[:,imp_features], df[self.dictionary['target'][0]])
            joblib.dump(model, self.dictionary['path']+'/'+'saved_objects/xgb_' + identifier)
            feature_imp = pd.DataFrame({'feature_names': imp_features, 'importance': model.feature_importances_})
            feature_imp.to_csv(self.dictionary['path']+'/'+'results/feature_importance_' + identifier + '.csv', index=False)
            score = model.predict_proba(df.loc[:,imp_features])
            if save_pred:
                out['pred'] = score[:,1]
                out.to_csv(self.dictionary['path']+'/'+'results/pred_dev_'+ identifier + '.csv', index=False)
            ks = self.ksTable(score[:,1],df[self.dictionary['target'][0]], 'dev_xgb_' + identifier)
            breaks = np.diff(ks['No.Res']) > 0
            dec_break = (np.diff(ks['No.Res']) > 0).any()
            ks_val = ks.KS.max()
            ks_decile = ks.KS.idxmax() + 1
            capture = ks['percent_cum_res'][3]
            if dec_break:
                break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
                summary_df = summary_df.append(pd.DataFrame([ list(row.values) + [ ks_val, break_dec, ks_decile, capture]],columns= list(row.index) + ['dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
            else:
                break_dec = np.nan
                summary_df = summary_df.append(pd.DataFrame([ list(row.values) + [ ks_val, break_dec, ks_decile, capture]],columns= list(row.index) + ['dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
            identifier = str(len(imp_features)) + 'var'
        summary_df.to_csv(self.dictionary['path']+'/'+'results/summary_df_params_xgb_' + version + '.csv', index =False)
        logging.debug("saveTrainingsDev module of XgbSelection Class executed successfully. summary is :{} ".format(summary_df))
        logging.debug(" dictionary is :{} ".format(self.dictionary))
    
    """ Gets the models built in saveTrainingsDev-modules and saves results on ITV/OTV
    for given feature combination.The metrics saved are:['itv/otv_ks', 
    'itv/otv_ro_break', 'itv/otv_ks_decile', 'itv/otv_capture',dev_itv/otv_ks_diff_perc]
    Parameters:
    ----------
    df : Input dataframe(itv/otv)
    imp_features : list of important features to build the model
    dset : dataset type for validation ('itv'/'otv')
    """                
    def saveTestingsVal(self,df,imp_features,dset, save_pred = False):
        logging.debug("inside saveTrainingsVal Module of XgbSelection Class .")
        logging.debug('Applying parameter iteration on  {0}'.format(dset))
        print('Applying parameter iteration on {0}'.format(dset))
        version=self.version
        params_df=self.params_df
        out = df.loc[:, self.dictionary['id'] + self.dictionary['performance']]
        out['actual'] = df[self.dictionary['target'][0]]
        summary_df_test =  pd.DataFrame()
        summary_df = pd.read_csv(self.dictionary['path']+'/'+'results/summary_df_params_xgb_' + version + '.csv')
        identifier = str(len(imp_features)) + 'var'
        alias = {
                'n_estimators': 'est',
                'max_depth': 'max_dep',
                'subsample': 'sub_s',
                'learning_rate': 'learn_r',
                'colsample_bytree': 'col_samp',
    	          'reg_lambda': 'lambda',
                  'gamma':'gamma',
                  'min_child_weight':'mcw'
    	           }
        for idx, row in params_df.astype(object).iterrows():
            print('Iteration {0} of {1}'.format(idx +1, params_df.shape[0]))
            tup = [i for i in zip([alias.get(row.index[j]) for j in range(len(params_df.columns))], row.values.astype(str))]
            params_str = [''.join(t) for t in tup]
            identifier = identifier +'_'.join(params_str) + '_' + version
            #param = row.to_dict()
            model = joblib.load(self.dictionary['path']+'/'+'saved_objects/xgb_' + identifier)
            score = model.predict_proba(df.loc[:,imp_features])
            if save_pred:
                out['pred'] = score[:,1]
                out.to_csv(self.dictionary['path']+'/'+'results/pred_' + dset + '_'+ identifier + '.csv', index=False)
            ks = self.ksTable(score[:,1],df[self.dictionary['target'][0]], dset + '_xgb_' + identifier)
            breaks = np.diff(ks['No.Res']) > 0
            dec_break = (np.diff(ks['No.Res']) > 0).any()
            ks_val = ks.KS.max()
            ks_decile = ks.KS.idxmax() + 1
            capture = ks['percent_cum_res'][3]
            if dec_break:
                break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
                summary_df_test = summary_df_test.append(pd.DataFrame([list(row.values) + [ks_val, break_dec, ks_decile, capture]], columns= list(row.index) + [dset + '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
            else:
                break_dec = np.nan
                summary_df_test = summary_df_test.append(pd.DataFrame([list(row.values) + [ks_val, break_dec, ks_decile, capture]], columns= list(row.index) + [dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
            identifier = str(len(imp_features)) + 'var'
        summary_df_test.reset_index(drop=True, inplace =True)
        summary_df[dset + '_ks'] = summary_df_test[dset + '_ks']
        summary_df[dset +'_ro_break'] = summary_df_test[dset +'_ro_break']
        summary_df[dset + '_ks_decile'] = summary_df_test[dset + '_ks_decile']
        summary_df[dset + '_capture'] = summary_df_test[dset + '_capture']
        summary_df['dev_' + dset + '_ks_diff'] = (summary_df['dev_ks'] - summary_df[dset + '_ks'])*100/summary_df['dev_ks']
        summary_df.to_csv(self.dictionary['path']+'/'+'results/summary_df_params_xgb_' + version + '.csv', index =False)
        logging.debug("saveTrainingsVal Module of XgbSelection Class executed successfully.")
    
    """ Saves summary of iterations on the paramters in the descending order of
        metric 'stability_weighted_otv_ks' and give ranks accordingly.
    
    """ 
    def XgbParamsIterationsSummary(self):
        logging.debug("inside XgbParamsIterationsSummary Module of XgbSelection Class .")
        iter_type='params'
        identifier='xgb_'+ self.version
        summary_df = pd.read_csv(self.dictionary['path']+'/'+'results/summary_df_' + iter_type + '_' + identifier + '.csv')
        summary_df['itv_otv_ks_diff'] = (summary_df['itv_ks'] - summary_df['otv_ks'])*100/summary_df['itv_ks']
        summary_df['dev_otv_diff_cat'] = np.where(summary_df['dev_otv_ks_diff'] <= 10, 1, 0)
        summary_df['otv_ro_cat'] = np.where(summary_df['otv_ro_break'].fillna(11) > 7, 1, 0)
        summary_df['itv_ro_cat'] = np.where(summary_df['itv_ro_break'].fillna(11) > 7, 1, 0)
        summary_df['dev_ro_cat'] = np.where(summary_df['dev_ro_break'].fillna(11) > 7, 1, 0)
        cols = ['dev_otv_diff_cat', 'otv_ro_cat', 'itv_ro_cat', 'dev_ro_cat']
        tups = summary_df[cols].sort_values(cols, ascending=False).apply(tuple, 1)
        f, i = pd.factorize(tups)
        factorized = pd.Series(f + 1, tups.index)
        summary_df = summary_df.assign(Rank1=factorized)
        
        tups2 = summary_df.loc[:,['Rank1', 'otv_ks']].sort_values(['Rank1', 'otv_ks'], ascending=[True, False]).apply(tuple, 1)
        f2, i2 = pd.factorize(tups2)
        factorized2 = pd.Series(f2 + 1, tups2.index)
        summary_df = summary_df.assign(Rank2 = factorized2)
        
        summary_df['dev_itv_ks_diff_score'] = 100 - abs(summary_df['dev_itv_ks_diff'])
        summary_df['dev_otv_ks_diff_score'] = 100 - abs(summary_df['dev_otv_ks_diff'])
        summary_df['itv_otv_ks_diff_score'] = 100 - abs(summary_df['itv_otv_ks_diff'])
        summary_df['dev_ro_score'] = 100*summary_df['dev_ro_break'].fillna(11)/11
        summary_df['itv_ro_score'] = 100*summary_df['itv_ro_break'].fillna(11)/11
        summary_df['otv_ro_score'] = 100*summary_df['otv_ro_break'].fillna(11)/11
        
        summary_df['stability_score'] = (summary_df['dev_itv_ks_diff_score'] + summary_df['dev_otv_ks_diff_score'] + summary_df['itv_otv_ks_diff_score'] + summary_df['dev_ro_score'] + summary_df['itv_ro_score'] + summary_df['otv_ro_score'])/6
        summary_df['stability_weighted_otv_ks'] = summary_df['stability_score'] * summary_df['otv_ks']
        
        summary_df.sort_values('stability_weighted_otv_ks', ascending=False, inplace=True)
        summary_df.to_csv(self.dictionary['path']+'/'+'results/summary_df_' + iter_type + '_' + identifier + '_ordered.csv', index=False)
        logging.debug(" XgbParamIterationsSumary Module of XgbSelection Class executed successfully. summary is :{} ".format(summary_df))
        return summary_df        
            
    
    '''Selects the parameters of the Xgb model based on the rank specified by the user
    '''

    def XgbParamsSelection(self,rank):
        logging.debug("inside XgbParamSelection Module of XgbSelection Class .rank selected by user is : {}".format(rank))
        version=self.version
        summary_df=pd.read_csv('results/summary_df_params_xgb_' + version + '_ordered.csv')
        params = ['n_estimators', 'max_depth', 'subsample', 'learning_rate', 'colsample_bytree', 'reg_lambda', 'reg_alpha','gamma','min_child_weight']
        param_cols = [i for i in summary_df.columns if i in params]
        final_params = summary_df[summary_df.Rank2== rank].iloc[0][param_cols].to_dict()
        final_params['n_estimators'] = int(final_params['n_estimators'])
        final_params['max_depth'] = int(final_params['max_depth'])
        final_params['subsample'] = round(final_params['subsample'],2)
        final_params['learning_rate'] = round(final_params['learning_rate'],2)
        final_params['colsample_bytree'] = round(final_params['colsample_bytree'],2)
        logging.debug("final parameters of XgbSelection are :{}".format(final_params))
        final_params = collections.OrderedDict(final_params)
        return final_params
		
    """ Reduces variables iteratively until no flat PDP is found
        
        Parameters:
        ----------
        devset: training set including response variable
        valsets: list of test sets including response variable
        valnames: list of corresponding names of validation sets, used in naming convention of result files
        feature_names: names of initial features from which reduction should start
        params: parameters for the model
    """
    ###################Tuple or list??
    #train, [itv, otv], ['itv', 'otv'], target_var, final_features, final_params, version, mypath
    def pdpVarReduction(self,devset, valsets, valnames,feature_names, params):
        logging.debug("inside pdpVarReduction Module of XgbSelection Class .")
        print('Reducing variable using partial dependency plots ..')
        y=self.dictionary['target'][0]
        version=self.version
        save_loc=self.dictionary['path']
        num_flat = len(feature_names)
        nonflat = feature_names
        summary_df_pdp = pd.DataFrame()
        dct= collections.OrderedDict(params)
        identifier='_'.join([list(dct.keys())[i] + str(list(dct.values())[i]) for i in range(len(dct.keys()))])
        X_train = devset
        
        while num_flat > 0:
            summary_df =  pd.DataFrame()
            curr_X = X_train[nonflat]
            target = X_train[y]
            #model = XGBClassifier(seed=10, **params, nthread=10)
            model = XGBClassifier(seed = 10,learning_rate=params['learning_rate'],colsample_bytree=params['colsample_bytree'],
                                  n_estimators=params['n_estimators'],subsample=params['subsample'],max_depth=params['max_depth'],
                                                    gamma=params['gamma'] ,min_child_weight=params['min_child_weight'],nthread = 10)
            model.fit(curr_X, target)
            joblib.dump(model, self.dictionary['path']+'/'+'saved_objects/xgb_nonflat_pdp_'+ version + '_' + identifier + '_' + str(len(nonflat)) + '.joblib', compress = 1)
            feature_imp = pd.DataFrame({'feature_names': nonflat, 'importance': model.feature_importances_})
            feature_imp.to_csv(self.dictionary['path']+'/'+'results/feature_importance_nonflat_pdp_'+ version + '_'  + identifier + '_' + str(len(nonflat)) + '.csv', index=False)
            score = model.predict_proba(curr_X)
            ks = self.ksTable(score[:,1], target, 'dev' + '_xgb_nonflat_pdp_' + version + '_'  + identifier + '_'+ str(len(nonflat)))
            breaks = np.diff(ks['No.Res']) > 0
            dec_break = (np.diff(ks['No.Res']) > 0).any()
            ks_val = ks.KS.max()
            ks_decile = ks.KS.idxmax() + 1
            #Top 3 decile capture
            capture = ks['percent_cum_res'][3]
            if dec_break:
                break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
                summary_df = summary_df.append(pd.DataFrame([[len(nonflat), ks_val, break_dec, ks_decile, capture]],columns=['feature_count', 'dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
            else:
                break_dec = np.nan
                summary_df = summary_df.append(pd.DataFrame([[len(nonflat), ks_val, break_dec, ks_decile, capture]],columns=['feature_count', 'dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
            
            for X_test, dset in zip(valsets, valnames):
                summary_df_test = pd.DataFrame()
                curr_X = X_test[nonflat]
                target = X_test[y]
                score = model.predict_proba(curr_X)
                ks = self.ksTable(score[:,1], target, dset + '_xgb_nonflat_pdp_' + version + '_'  + identifier + '_' + str(len(nonflat)))
                breaks = np.diff(ks['No.Res']) > 0
                dec_break = (np.diff(ks['No.Res']) > 0).any()
                ks_val = ks.KS.max()
                ks_decile = ks.KS.idxmax() + 1
                capture = ks['percent_cum_res'][3]
                if dec_break:
                    break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
                    summary_df_test = summary_df_test.append(pd.DataFrame([[len(nonflat), ks_val, break_dec, ks_decile, capture]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
                else:
                    break_dec = np.nan
                    summary_df_test = summary_df_test.append(pd.DataFrame([[len(nonflat), ks_val, break_dec, ks_decile, capture]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
                
                summary_df_test.reset_index(drop=True, inplace =True)
                summary_df[dset + '_ks'] = summary_df_test[dset + '_ks']
                summary_df[dset +'_ro_break'] = summary_df_test[dset +'_ro_break']
                summary_df[dset + '_ks_decile'] = summary_df_test[dset + '_ks_decile']
                summary_df[dset + '_capture'] = summary_df_test[dset + '_capture']
                summary_df['dev_' + dset + '_ks_diff'] = (summary_df['dev_ks'] - summary_df[dset + '_ks'])*100/summary_df['dev_ks']
                
            summary_df_pdp = summary_df_pdp.append(summary_df)
        
            nonflat_prev = nonflat
            if not os.path.exists(self.dictionary['path']+'/'+'PDP/' + version + '_'  + identifier + '_' + str(len(nonflat))):
                os.makedirs(self.dictionary['path']+'/'+'PDP/' + version + '_'  + identifier + '_' + str(len(nonflat)))
            nonflat = self.generatePDP(model, X_train, nonflat, os.path.join(save_loc, self.dictionary['path']+'/'+'PDP/' + version + '_'  + identifier + '_' + str(len(nonflat))))
            num_flat = len(set(nonflat_prev)-set(nonflat))
        summary_df_pdp.to_csv(self.dictionary['path']+'/'+'results/summary_df_nonflat_pdp_xgb_' + version + '_'  + identifier + '.csv', index=False)
        logging.debug("pdpvarreduction Module of XgbSelection Class executed successfully.")
        return nonflat
    ####Response as 1 and 0
    def ksTable(self,score, response, identifier):
        logging.debug("inside KsTable Module of XgbSelection Class .score is :{}   , response is :{} , identifier is : {}".format(score,response,identifier))
        print('getting KS..')
        group = 10
        df = pd.DataFrame({'score': score, 'response' : response})
        df = df.sort_values(['score'], ascending = [False])
        bin_size = len(score)/group
        rem = len(score) % group
        if rem == 0:
            df['groups'] = list(np.repeat(range(rem+1,11), bin_size))
        else:
            df['groups'] = list(np.repeat(range(1,rem+1),bin_size + 1)) + list(np.repeat(range(rem+1,11), bin_size))
        grouped = df.groupby('groups', as_index =False)
        agg = pd.DataFrame({'Total_Obs': grouped.count().response})
        agg['No.Res'] = grouped.sum().response
        agg['No.Non_Res'] = agg['Total_Obs'] - agg['No.Res']
        agg['min_pred'] = grouped.min().score
        agg['max_pred'] = grouped.max().score
        agg['pred_rr'] = grouped.mean().score
        agg['cum_no_res'] = agg['No.Res'].cumsum()
        agg['cum_no_non_res'] = agg['No.Non_Res'].cumsum()
        agg['percent_cum_res'] = agg['cum_no_res']/agg['cum_no_res'].max()
        agg['percent_cum_non_res'] = agg['cum_no_non_res']/agg['cum_no_non_res'].max()
        agg['KS'] = agg['percent_cum_res'] - agg['percent_cum_non_res']
        agg.to_csv(self.dictionary['path']+'/'+'results/KS_table_'+ identifier + '.csv', index = False)
        logging.debug("KS Table is : {}".format(agg))
        return(agg)
    
    
    
    """ Creates partial dependence plots
        
        Parameters:
        ----------
        model: fitted model object for which partial dependence plots needs to be drawn
        X: dataset with the independent variables
        features: index of features in X for which partial dependence has to be calculated
        n_cols: number of columns in the plot grid
        figsize: overall plot size in inches
    """    
            
    def plotPartialDependence(self,model, X, features, n_cols=3, figsize=(10, 10)):
        logging.debug("inside plotPartialDependence Module of XgbSelection Class .")
        fig = plt.figure(figsize=figsize)
        nrows=int(np.ceil(len(features)/float(n_cols)))
        ncols=min(n_cols, len(features))
        axs = []
        pdp = []
        for i, f_id in enumerate(features):
            X_temp = X.copy().values
            ax = fig.add_subplot(nrows, ncols, i + 1)
            
            x_scan = np.linspace(np.percentile(X_temp[:, f_id], 0.1), np.percentile(X_temp[:, f_id], 99.5), 10)
            y_partial = []
            
            for point in x_scan:
                X_temp[:, f_id] = point
                X_temp_df = pd.DataFrame(X_temp, columns=X.columns)
                pred = model.predict_proba(X_temp_df)
                y_partial.append(np.average(pred[:,1]))
            
            y_partial = np.array(y_partial)
            pdp.append((x_scan, y_partial))
            
            # Plot partial dependence
            ax.plot(x_scan, y_partial, '-', color = 'green', linewidth = 1)
            ax.set_xlim(min(x_scan)-0.1*(max(x_scan)-min(x_scan)), max(x_scan)+0.1*(max(x_scan)-min(x_scan)))
            ax.set_ylim(min(y_partial)-0.1*(max(y_partial)-min(y_partial)), max(y_partial)+0.1*(max(y_partial)-min(y_partial)))
            ax.set_xlabel(X.columns[f_id])
        axs.append(ax)
        fig.subplots_adjust(bottom=0.15, top=0.7, left=0.1, right=0.95, wspace=0.6,
                            hspace=0.3)
        fig.tight_layout()
        logging.debug("plotPArtialDependece Module of XgbSelection Class executed successfully.")
        return fig, axs, pdp
  
    """ Generates partial dependence plots for all variables in a dataset iteratively
        
        Parameters:
        ----------
        model: fitted model object for which partial dependence plots needs to be drawn
        X: dataset with the independent variables
        feature_names: name of all features for which PDP has to be drawn
        save_loc: save location path for the PDP images
    """
    
    def generatePDP(self,model, X, feature_names, save_loc):
        logging.debug("inside generatePDP Module of XgbSelection Class .")
        num_features = len(feature_names)
        plots_per_grid = 9
        nonflat = []
        
        for i in range(int(np.ceil(num_features/plots_per_grid))):
            features = range(plots_per_grid*i, min(plots_per_grid*(i+1), num_features))
            fig, axs, pdp = self.plotPartialDependence(model, X[feature_names], features, figsize=(10, 10))
            fig.savefig(os.path.join(save_loc, str(i)+'.png'), dpi = 200)
            
            for f, p in zip (features, pdp):
                if max(p[1]) - min(p[1]) > 0:
                    nonflat.append(feature_names[f])
        logging.debug("generatePDP Module of XgbSelection Class executed successfully.")
        return nonflat 
    
    """ Creates Final Report
        
        Parameters:
        ----------
        final_params: Dictionary with key as XGBoost parameters and value as tuned parameter value
    	 final_features: List of features after pdp var reduction
        X_otv: OTV dataset to score on
    """
    #final_params,version,otv
    def finalReport(self,final_params,final_features,X_otv,X_itv):
        logging.debug("inside finalReport Module of XgbSelection Class .")
        nonflat=final_features
        version=self.version
        final_params= collections.OrderedDict(final_params)
        #print('error in next line') 
        identifier='_'.join([list(final_params.keys())[i] + str(list(final_params.values())[i]) for i in range(len(final_params.keys()))])
        gbm = joblib.load(self.dictionary['path']+'/'+'saved_objects/xgb_nonflat_pdp_' + version + '_'+ identifier + '_'+ str(len(nonflat))+ '.joblib')
        
        imp_features_df = pd.read_csv(self.dictionary['path']+'/'+'results/feature_importance_' + version + '.csv')
        feature_summary = pd.read_csv(self.dictionary['path']+'/'+'results/summary_df_features_xgb_' + version + '_ordered.csv')
        param_summary = pd.read_csv(self.dictionary['path']+'/'+'results/summary_df_params_xgb_' + version + '_ordered.csv')
        pdp_summary = pd.read_csv(self.dictionary['path']+'/'+'results/summary_df_nonflat_pdp_xgb_' + version + '_' + identifier +'.csv')
        os.chdir(self.dictionary['path'])
        writer = ExcelWriter('xgb_final_reports_'+ version +'.xlsx')
        #writer = ExcelWriter(self.dictionary['path']+'/'+'xgb_final_reports_'+ version +'.xlsx')
        #os.chdir(self.dictionary['path'])
        #feature_summary.to_csv(self.dictionary['path']+'/'+'Final_feature_summary.csv', index = False)
        imp_features_df.to_excel(writer,'Initial Model Variables', index =False)
        feature_summary.to_excel(writer,'feature_summary', index =False)
        #param_summary.to_csv(self.dictionary['path']+'/'+'Final_parameter_summary.csv', index = False)
        
        param_summary.to_excel(writer,'parameter_summary', index =False)
        #pdp_summary.to_csv(self.dictionary['path']+'/'+'Final_pdp_summary.csv', index = False)
        pdp_summary.to_excel(writer,'pdp_summary', index =False)
       
        feature_names = pd.read_csv(self.dictionary['path']+'/'+'results/feature_importance_nonflat_pdp_' + version +'_' + identifier + '_' + str(len(nonflat)) + '.csv')
        #feature_names.to_csv(self.dictionary['path']+'/'+'Final_feature_importance.csv', index = False)
        feature_names.to_excel(writer,'feature_importance', index =False)
        feature_names = feature_names.feature_names.tolist()
        
        dev_ks = pd.read_csv(self.dictionary['path']+'/'+'results/KS_table_dev_xgb_nonflat_pdp_' + version + '_' + identifier + '_' + str(len(nonflat)) + '.csv')
        itv_ks = pd.read_csv(self.dictionary['path']+'/'+'results/KS_table_itv_xgb_nonflat_pdp_' + version + '_' + identifier + '_' + str(len(final_features))  + '.csv')
        otv_ks = pd.read_csv(self.dictionary['path']+'/'+'results/KS_table_otv_xgb_nonflat_pdp_' + version + '_' + identifier + '_' + str(len(final_features))  + '.csv')
        #dev_ks.to_csv(self.dictionary['path']+'/'+'Final_dev_ks.csv', index = False)
        dev_ks.to_excel(writer,'dev_ks', index =False)
        #itv_ks.to_csv(self.dictionary['path']+'/'+'Final_itv_ks.csv', index = False)
        itv_ks.to_excel(writer,'itv_ks', index =False)
        #otv_ks.to_csv(self.dictionary['path']+'/'+'Final_otv_ks.csv', index = False)
        otv_ks.to_excel(writer,'otv_ks', index =False)
        
        otv = X_otv.loc[:,feature_names]
        #itv = X_itv.loc[:,feature_names]
        score = gbm.predict_proba(otv)
        otv['score'] = score[:,1]
        otv['dev_grps'] = np.where(otv.score > dev_ks.min_pred[0], 0, np.where(otv.score > dev_ks.min_pred[1], 1, np.where(otv.score > dev_ks.min_pred[2], 2, np.where(otv.score > dev_ks.min_pred[3], 3, np.where(otv.score > dev_ks.min_pred[4], 4, np.where(otv.score > dev_ks.min_pred[5], 5, np.where(otv.score > dev_ks.min_pred[6], 6, np.where(otv.score > dev_ks.min_pred[7], 7, np.where(otv.score > dev_ks.min_pred[8], 8, np.where(otv.score <= dev_ks.min_pred[8], 9,-99))))))))))
        
        grouped = otv.groupby('dev_grps')
        agg = pd.DataFrame({'Total_Obs': grouped.count().score})
        agg['percent_obs'] = agg['Total_Obs']*100/agg['Total_Obs'].sum()
        
        agg['psi'] = (0.1 - agg['percent_obs']/100) * np.log(0.1/(agg['percent_obs']/100))
        
        print('DEV-OTV PSI value is {}'.format(agg['psi'].sum()))
        #agg.to_csv(self.dictionary['path']+'/'+'Final_DEV_OTV_PSI.csv', index = False)
        agg.to_excel(writer,'DEV_OTV_PSI', index =False)
        logging.debug("DEV-OTV PSI value is  {}".format(agg['psi'].sum()))
        logging.debug('DEV-OTV PSI table is as follows  {}'.format(agg['psi']))
        del agg
        #writer.save()
        #otv['score'] = score[:,1]
        otv['itv_grps'] = np.where(otv.score > itv_ks.min_pred[0], 0, np.where(otv.score > itv_ks.min_pred[1], 1, np.where(otv.score > itv_ks.min_pred[2], 2, np.where(otv.score > itv_ks.min_pred[3], 3, np.where(otv.score > itv_ks.min_pred[4], 4, np.where(otv.score > itv_ks.min_pred[5], 5, np.where(otv.score > itv_ks.min_pred[6], 6, np.where(otv.score > itv_ks.min_pred[7], 7, np.where(otv.score > itv_ks.min_pred[8], 8, np.where(otv.score <= itv_ks.min_pred[8], 9,-99))))))))))
        
        grouped = otv.groupby('itv_grps')
        agg = pd.DataFrame({'Total_Obs': grouped.count().score})
        agg['percent_obs'] = agg['Total_Obs']*100/agg['Total_Obs'].sum()
        
        agg['psi'] = (0.1 - agg['percent_obs']/100) * np.log(0.1/(agg['percent_obs']/100))
        
        print('ITV-OTV PSI value is {}'.format(agg['psi'].sum()))
        #agg.to_csv(self.dictionary['path']+'/'+'Final_ITV_OTV_PSI.csv', index = False)
        agg.to_excel(writer,'ITV_OTV_PSI', index =False)
        logging.debug("ITV-OTV PSI value is  {}".format(agg['psi'].sum()))
        logging.debug('ITV-OTV PSI table is as follows  {}'.format(agg['psi']))
        del agg
        print('Final reports file ' + 'xgb_final_reports_'+ version +'.xlsx' + '_is created at path: ', self.dictionary['path'])
        #otv = X_otv.loc[:,feature_names]
        itv = X_itv.loc[:,feature_names]
        score = gbm.predict_proba(itv)
        itv['score'] = score[:,1]
        itv['dev_grps'] = np.where(itv.score > dev_ks.min_pred[0], 0, np.where(itv.score > dev_ks.min_pred[1], 1, np.where(itv.score > dev_ks.min_pred[2], 2, np.where(itv.score > dev_ks.min_pred[3], 3, np.where(itv.score > dev_ks.min_pred[4], 4, np.where(itv.score > dev_ks.min_pred[5], 5, np.where(itv.score > dev_ks.min_pred[6], 6, np.where(itv.score > dev_ks.min_pred[7], 7, np.where(itv.score > dev_ks.min_pred[8], 8, np.where(itv.score <= dev_ks.min_pred[8], 9,-99))))))))))
        
        grouped = itv.groupby('dev_grps')
        agg = pd.DataFrame({'Total_Obs': grouped.count().score})
        agg['percent_obs'] = agg['Total_Obs']*100/agg['Total_Obs'].sum()
        
        agg['psi'] = (0.1 - agg['percent_obs']/100) * np.log(0.1/(agg['percent_obs']/100))
        
        print('DEV-ITV PSI value is {}'.format(agg['psi'].sum()))
        #agg.to_csv(self.dictionary['path']+'/'+'Final_DEV-ITV_PSI.csv', index = False)
        agg.to_excel(writer,'DEV_ITV_PSI', index =False)
        logging.debug("DEV-ITV PSI value is  {}".format(agg['psi'].sum()))
        logging.debug('DEV-ITV PSI table is as follows  {}'.format(agg['psi']))
        writer.save()

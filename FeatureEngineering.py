"""
Version v0.9.2

"""
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta
from xgboost import XGBClassifier
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import euclidean_distances
import logging
""" Feature Engineering Class """

class FeatureEngineering:
    """
    Parameters  : 
        
    """
    
    def __init__(self,desc_dict,version):
        logging.debug("inside Feature engineering Class Constructor. dictionary : {} and Version: {}".format(desc_dict,version))
        self.df=None
        self.dictionary=desc_dict
        self.version=str(version)
        if not os.path.exists('results'):
            os.makedirs('results')
 
        if not os.path.exists('saved_objects'):
            os.makedirs('saved_objects')
 
        if not os.path.exists('PDP'):
            os.makedirs('PDP')
        

    def periodDeltaPast(self,timestamp,delta,delta_type):
        logging.debug("inside periodDeltaPAst Module of Feature Engineering.")
        #td = np.timedelta64(delta, delta_type)
        timestamp=timestamp.to_pydatetime()
        
        if delta_type=='d':
            new_timestamp= timestamp - dateutil.relativedelta.relativedelta(days=delta)
        elif delta_type=='M':
            new_timestamp= timestamp - dateutil.relativedelta.relativedelta(months=delta)
        return pd.to_datetime(new_timestamp)
        
    def periodDeltaFuture(self,timestamp,delta,delta_type):
        logging.debug("inside periodDeltaFuture Module of Feature Engineering.")
        td = np.timedelta64(delta, delta_type)
        if delta_type != 'h':
            new_timestamp=pd.Timestamp((timestamp - td).date())
        else:
            new_timestamp= timestamp + td
        return new_timestamp
    
    
    def createCentralityFeatures(self,df,cohort_period_type=None,centrality_period=None,centrality_order=None,split_type=None):
        logging.debug("inside Centality Feature Creation Module of Feature Engineering class. cohort_period_type={} , centrality_perion={} , centrality_order={} , split_type={}".format(cohort_period_type, centrality_period, centrality_order , split_type ))
        if(split_type=="dev"):
            if(self.dictionary['dev_cohort']==None):
                raise Exception("Dev_cohort was not defined in the beginning , please make sure you have provided all the detailes in the beginning if not then restart the process again")
        elif(split_type=="otv"):
            if(self.dictionary['otv_cohort']==None):
                raise Exception("Dev_cohort was not defined in the beginning , please make sure you have provided all the detailes in the beginning if not then restart the process again")
        
        self.df=df
        self.dictionary['centrality_period']=centrality_period
        self.dictionary['cohort_period_type']=cohort_period_type
        self.dictionary['centrality_order']=centrality_order
        self.dictionary['split_type']=split_type
        windows=[centrality_period+ order*centrality_period for order in range(centrality_order)]

        if(split_type=="itv"):
            cohort=self.dictionary['cohort_df']['cohort']["dev"]
        else:
            cohort=self.dictionary['cohort_df']['cohort'][split_type]

        df_temp=self.df.loc[:,self.dictionary['id']+self.dictionary['performance']+self.dictionary['num']]
        periods=[0 for i in range(0,len(windows))]
        out_df = self.df[self.df[self.dictionary['performance'][0]]==cohort].loc[:,self.dictionary['id']+self.dictionary['performance']]
        centrality_features_list=[]
        print("creating centrality features ...")
        for i in range(0,len(windows)):
            periods[i]=[self.periodDeltaPast(cohort,delta,cohort_period_type) for delta in range(windows[i])]
            df_temp2=df_temp[df_temp[self.dictionary['performance'][0]].isin(periods[i])]
            df_temp3=df_temp2.groupby(self.dictionary['id'],as_index=False)[self.dictionary['num']].agg({'min', 'max', 'median', 'mean'})
            df_temp3.columns = df_temp3.columns.map(('_'+str(windows[i])).join)
            centrality_features_list=centrality_features_list+df_temp3.columns.tolist()
            df_temp3.reset_index(inplace=True)
            out_df=out_df.merge(df_temp3,on=self.dictionary['id'],how='left')
        self.dictionary['centrality_features']=centrality_features_list
        logging.debug("Centrality features are created successfully. List of Centrality features : {}. dictionary is: {}".format(self.dictionary['centrality_features'] , self.dictionary))
        return out_df,self.dictionary
      
      
    def vec_dist(self,df,col):
        logging.debug("inside vec_dict module of Feature Engineering class.")
        out = pd.DataFrame(index = df.index)
        mean_vec = df.mean(axis = 0)
        mean_vec = np.array([mean_vec])
        dist = euclidean_distances(df, mean_vec)
        out[col + '_euclidean_ dist'] = dist[:,0]
        out[col + '_cosine_sim'] = df.apply(lambda x: cosine(x.values, mean_vec), axis = 1)
        out[col + '_corr']=df.apply(lambda x: np.corrcoef(x.values, mean_vec)[0,1],axis=1)
        return out

    
    def vec_count(self,df,col):
        logging.debug("inside vec_count Module of Feature Engineering.")
        mean_vec = df.mean(axis = 0)
        mean_vec = np.array(mean_vec)
        diff = df.sub(mean_vec)
        diff = diff > 0
        diff_count = pd.DataFrame(diff.sum(axis =1))
        diff_count.columns = [col + '_diff_count']
        return diff_count  
    
    def createSequenceFeature(self,df,cohort_period_type=None,n_month=12,split_type=None):
        logging.debug("inside Sequence Feature Creation Module of Feature Engineering. arguments are - >  cohort_period_type:{} , n_month={} , split_Type={}".format(cohort_period_type,n_month,split_type))
        if(split_type=="dev"):
            if(self.dictionary['dev_cohort']==None):
                raise Exception("Dev_cohort was not defined in the beginning , please make sure you have provided all the detailes in the beginning if not then restart the process again")
        elif(split_type=="otv"):
            if(self.dictionary['otv_cohort']==None):
                raise Exception("Dev_cohort was not defined in the beginning , please make sure you have provided all the detailes in the beginning if not then restart the process again")

        if cohort_period_type==None:
            cohort_period_type=self.cohort_period_type
        else:
            self.cohort_period_type=cohort_period_type
            
        self.dictionary['split_type']=split_type
        if(split_type=="itv"):
            cohort=self.dictionary['cohort_df']['cohort']["dev"]
        else:
            cohort=self.dictionary['cohort_df']['cohort'][split_type]
            
        nmths=[self.periodDeltaPast(cohort,delta,cohort_period_type) for delta in range(n_month)]
        coh_df = df[df[self.dictionary['performance'][0]].isin(nmths)]
        coh_df.sort_values(self.dictionary['id'] + self.dictionary['performance'], ascending = [True]*len(self.dictionary['id']) + [True], inplace = True)
        coh_df.set_index(self.dictionary['id'] + self.dictionary['performance'], inplace =True)
        coh_df = coh_df.loc[:,self.dictionary['seq']]
        coh_df = coh_df.unstack()
        out_df = pd.DataFrame()
        print('Creating sequence features..')
        for i in self.dictionary['seq']:
            temp_df = coh_df.loc[:,i]
            temp_df_t = temp_df.transpose()
            mean_val = temp_df_t.mean()
            temp_df_t = temp_df_t.fillna(mean_val)
            temp_df = temp_df_t.transpose()
            #print(i)
            #print(temp_df.head())
            dist_df = self.vec_dist(temp_df,i)
            diff_count = self.vec_count(temp_df, i)
            out_df = pd.concat([out_df, dist_df, diff_count], axis = 1)
        out_df.reset_index(inplace =True)
        out_df[self.dictionary['performance'][0]] = cohort
        self.dictionary['sequence_features']=out_df.columns
        logging.debug("Created Sequence Feature are : {} and the dictionary is : {}".format(self.dictionary['sequence_features'],self.dictionary))
        return out_df,self.dictionary
    
    def acc_fts(self,df,id_vars,ft_vars,delta_periods):
        logging.debug("inside acc_fts Module of Feature Engineering.")
        print('Creating acceleration features..')
        for count in range(1,len(delta_periods) - 1):
            for i in range(len(ft_vars)):
                df[ft_vars[i]+ '_acc_' + str(count)]=df.iloc[:,i + len(id_vars) + 1 +4*(count-1)]-df.iloc[:,i + len(id_vars) + 1 +4*(count-1) +2*len(ft_vars)]
        return df

    
    """ Returns a data frame with new velocity and acceleration features
        
        Parameters:
        ----------
        df : Input dataframe with id_vars, time_vars, ft_vars 
        id_vars : List of ID columns in the dataframe df. (Generally CLNT_NBR or CRD_ACCT_NBR etc)
        time_var: String column name of the time variable (PERFORMANCE_PERIOD)
        ft_vars: List of variables which are time varying continuous values. Velocity features will be created for these variables
        cohort: Cohort year month. Format accepted YYYYMM e.g 201802
        n_month: Number of months of data to use for sequence features. Default to previous 12 months
    """

    def velocity_fts(self,df, cohort_period_type=None, n_month = 12,split_type=None):
        logging.debug("inside velocity Feature Creation Module of Feature Engineering.")
        
        if(split_type=="dev"):
            if(self.dictionary['dev_cohort']==None):
                raise Exception("Dev_cohort was not defined in the beginning , please make sure you have provided all the detailes in the beginning if not then restart the process again")
        elif(split_type=="otv"):
            if(self.dictionary['otv_cohort']==None):
                raise Exception("Dev_cohort was not defined in the beginning , please make sure you have provided all the detailes in the beginning if not then restart the process again")

        if cohort_period_type==None:
            cohort_period_type=self.cohort_period_type
        else:
            self.cohort_period_type=cohort_period_type
        self.dictionary['split_type']=split_type
        if(split_type=="itv"):
            cohort=self.dictionary['cohort_df']['cohort']["dev"]    
        else:
            cohort=self.dictionary['cohort_df']['cohort'][split_type]
        delta_list = list(range(n_month, 0, -3))
        out = df[df[self.dictionary['performance'][0]] == cohort].loc[:, self.dictionary['id'] + self.dictionary['performance']]
        delta_periods = [self.periodDeltaPast(cohort, dm,cohort_period_type) for dm in delta_list] + [cohort]
        df.sort_values(self.dictionary['id'] + self.dictionary['performance'], ascending = [True]*len(self.dictionary['id']) + [False], inplace = True)
        a= len(delta_periods) - 1
        print('Creating velocity features..')
        for i in range(a):
            temp_df = df[df[self.dictionary['performance'][0]].isin(delta_periods[i:i+2])]
            diff_fts = pd.concat([temp_df[self.dictionary['id']], temp_df[self.dictionary['num']].diff()], axis=1)
            diff_fts = diff_fts[(diff_fts.loc[:,self.dictionary['id']].shift(1) == diff_fts.loc[:,self.dictionary['id']]).all(axis =1)]
            #print diff_fts
            ratio_fts = pd.concat([temp_df[self.dictionary['id']], temp_df[self.dictionary['num']].pct_change()], axis=1)
            ratio_fts = ratio_fts[(ratio_fts.loc[:,self.dictionary['id']].shift(1) == ratio_fts.loc[:,self.dictionary['id']]).all(axis =1)]
            if i < (len(delta_list) - 1):
                end_m = str(-1*delta_list[i+1]) 
            else:
                end_m = 'curr_'
            diff_fts.columns = self.dictionary['id'] + [col + '_' + str(-1*delta_list[i]) + 'm_to_' + end_m + 'm_diff' for col in self.dictionary['num']]
            diff_fts.dropna(inplace = True)
            ratio_fts.columns = self.dictionary['id'] +  [col + '_' + str(-1*delta_list[i]) + 'm_to_' + end_m + 'm_ratio' for col in self.dictionary['num']]
            ratio_fts.dropna(inplace = True)
            out = out.merge(diff_fts, on = self.dictionary['id'], how =  'left')
            out = out.merge(ratio_fts, on = self.dictionary['id'], how =  'left')
        acc=self.acc_fts(out,self.dictionary['id'],self.dictionary['num'],delta_periods)
        self.dictionary['velocity_features']=acc.columns
        logging.debug("Created velocity features are : {} and dictionary is :{}".format(self.dictionary['velocity_features'],self.dictionary))
        return acc,self.dictionary
    
    
        
    def mergeRawAndCreatedFeatures(self,raw_df,list_created_ftrs_dfs):
        logging.debug("inside merge Raw and Created Features Module of Feature Engineering.")
        from functools import reduce
        dfs = [raw_df]+list_created_ftrs_dfs
        merged_df = reduce(lambda left, right: pd.merge(left, right, on = self.dictionary['id'] + self.dictionary['performance'], how = 'left'), dfs) 
        logging.debug("Merged dataframe with columns are : {}".format(merged_df.columns))
        return merged_df
        
        
    def createFeaturesWide(self,df_temp,seqvar , seqstart , seq_end):
        for i in range(len(seqvar)):
            if seqvar[i]!=99999:
                var=seqvar[i]
                start=seqstart[i]
                end=seq_end[i]
                j=start
                if(start<end and end >=3):
                    list_3=[]
                    j=start
                    for j in range(j,j+3):
                        list_3.append(var+"-"+str(j))
                    df_temp[var+str(j)+"_3mean"]=df_temp[list_3].mean(axis=1)
                    df_temp[var+str(j)+"_3sum"]=df_temp[list_3].sum(axis=1)
                    df_temp[var+str(j)+"_3std"]=df_temp[list_3].std(axis=1)
                    df_temp[var+str(j)+"_3median"]=df_temp[list_3].median(axis=1)
                    df_temp[var+str(j)+"_3min"]=df_temp[list_3].min(axis=1)
                    df_temp[var+str(j)+"_3max"]=df_temp[list_3].max(axis=1)
                    df_temp[var+str(j)+"_3diff"]=df_temp[var+"-"+str(start)] - df_temp[var+"-"+str(start+2)]
                    df_temp[var+str(j)+"_3ratio"]=df_temp[var+"-"+str(start)] / df_temp[var+"-"+str(start+2)]
                
                if(start<end and end>=6):
                    list_6=[]
                    j=start
                    for j in range(j,j+6):
                        list_6.append(var+"-"+str(j))
                    
                    df_temp[var+str(j)+"_6mean"]=df_temp[list_6].mean(axis=1)
                    df_temp[var+str(j)+"_6sum"]=df_temp[list_6].sum(axis=1)
                    df_temp[var+str(j)+"_6std"]=df_temp[list_6].std(axis=1)
                    df_temp[var+str(j)+"_6median"]=df_temp[list_6].median(axis=1)
                    df_temp[var+str(j)+"_6min"]=df_temp[list_6].min(axis=1)
                    df_temp[var+str(j)+"_6max"]=df_temp[list_6].max(axis=1)
                    df_temp[var+str(j)+"_6diff"]=df_temp[var+"-"+str(start)] - df_temp[var+"-"+str(start+5)]
                    df_temp[var+str(j)+"_6ratio"]=df_temp[var+"-"+str(start)] / df_temp[var+"-"+str(start+5)]
                
                if(start<end and end>=9):
                    list_9=[]
                    j=start
                    for j in range(j,j+9):
                        list_9.append(var+"-"+str(j))
                    
                    df_temp[var+str(j)+"_9mean"]=df_temp[list_9].mean(axis=1)
                    df_temp[var+str(j)+"_9sum"]=df_temp[list_9].sum(axis=1)
                    df_temp[var+str(j)+"_9std"]=df_temp[list_9].std(axis=1)
                    df_temp[var+str(j)+"_9median"]=df_temp[list_9].median(axis=1)
                    df_temp[var+str(j)+"_9min"]=df_temp[list_9].min(axis=1)
                    df_temp[var+str(j)+"_9max"]=df_temp[list_9].max(axis=1)
                    df_temp[var+str(j)+"_9diff"]=df_temp[var+"-"+str(start)] - df_temp[var+"-"+str(start+8)]
                    df_temp[var+str(j)+"_9ratio"]=df_temp[var+"-"+str(start)] / df_temp[var+"-"+str(start+8)]
                
                if(start<end and end>=12):
                    list_12=[]
                    j=start
                    for j in range(j,j+12):
                        list_12.append(var+"-"+str(j));
                    
                    df_temp[var+str(j)+"_12mean"]=df_temp[list_12].mean(axis=1)
                    df_temp[var+str(j)+"_12sum"]=df_temp[list_12].sum(axis=1)
                    df_temp[var+str(j)+"_12std"]=df_temp[list_12].std(axis=1)
                    df_temp[var+str(j)+"_12median"]=df_temp[list_12].median(axis=1)
                    df_temp[var+str(j)+"_12min"]=df_temp[list_12].min(axis=1)
                    df_temp[var+str(j)+"_12max"]=df_temp[list_12].max(axis=1)
                    df_temp[var+str(j)+"_12diff"]=df_temp[var+"-"+str(start)] - df_temp[var+"-"+str(start+11)]
                    df_temp[var+str(j)+"_12ratio"]=df_temp[var+"-"+str(start)] / df_temp[var+"-"+str(start+11)]
    
                if(start<end and end>=18):
                    list_18=[]
                    j=start
                    for j in range(j,j+18):
                        list_18.append(var+"-"+str(j));
                    df_temp[var+str(j)+"_18mean"]=df_temp[list_18].mean(axis=1)
                    df_temp[var+str(j)+"_18sum"]=df_temp[list_18].sum(axis=1)
                    df_temp[var+str(j)+"_18std"]=df_temp[list_18].std(axis=1)
                    df_temp[var+str(j)+"_18median"]=df_temp[list_18].median(axis=1)
                    df_temp[var+str(j)+"_18min"]=df_temp[list_18].min(axis=1)
                    df_temp[var+str(j)+"_18max"]=df_temp[list_18].max(axis=1)
                    df_temp[var+str(j)+"_18diff"]=df_temp[var+"-"+str(start)] - df_temp[var+"-"+str(start+17)]
                    df_temp[var+str(j)+"_18ratio"]=df_temp[var+"-"+str(start)] / df_temp[var+"-"+str(start+17)]
    
                if(start<end and end>=24):
                    list_24=[]
                    j=start
                    for j in range(j,j+24):
                        list_24.append(var+"-"+str(j));
                    df_temp[var+str(j)+"_24mean"]=df_temp[list_24].mean(axis=1)
                    df_temp[var+str(j)+"_24sum"]=df_temp[list_24].sum(axis=1)
                    df_temp[var+str(j)+"_24std"]=df_temp[list_24].std(axis=1)
                    df_temp[var+str(j)+"_24median"]=df_temp[list_24].median(axis=1)
                    df_temp[var+str(j)+"_24min"]=df_temp[list_24].min(axis=1)
                    df_temp[var+str(j)+"_24max"]=df_temp[list_24].max(axis=1)
                    df_temp[var+str(j)+"_24diff"]=df_temp[var+"-"+str(start)] - df_temp[var+"-"+str(start+23)]
                    df_temp[var+str(j)+"_24ratio"]=df_temp[var+"-"+str(start)] / df_temp[var+"-"+str(start+23)]
                
        return df_temp
    
    def saveFeatureIterationsDev(self,df,save_pred = False):
        logging.debug("inside saveFeatureIterationDev Module of Feature Engineering.")
        id_var=self.dictionary['id']
        time_var=self.dictionary['performance']
        y=self.dictionary['target']
        version=self.version
        logging.debug("First trial run with all the features")
        print('\nFirst trial run with all the features...')
        out = df.loc[:, id_var + time_var]
        out['actual'] = df[y[0]]
        model = XGBClassifier(seed=10)
        drop_vars = id_var + time_var + y #+self.dictionary['cat']
        logging.debug("Drop variables are : {}".format(drop_vars))
        model.fit(df.drop(drop_vars, axis=1), df[y[0]])
        imp_features_df = pd.DataFrame({'feature_names': df.drop(drop_vars, axis=1).columns, 'importance': model.feature_importances_})
        imp_features_df.sort_values('importance', ascending=False, inplace=True)
        imp_features_df.to_csv(self.dictionary['path']+'/'+'results/feature_importance_' + version + '.csv', index=False)
        
        if save_pred:
            pred = model.predict_proba(df.drop(drop_vars, axis=1))
            out['pred'] = pred
            out.to_csv(self.dictionary['path']+'/'+'results/pred_dev_overall_'+ version + '.csv', index=False)
            
        logging.debug("Iterating on different feature combination in dev...")
        
        print('Iterating on different feature combination in dev..')
        summary_df =  pd.DataFrame()
        imp_features_df = pd.read_csv(self.dictionary['path']+'/'+'results/feature_importance_' + version + '.csv')
        imp_features_df = imp_features_df.sort_values('importance', ascending = False)
        imp_features = imp_features_df[imp_features_df['importance'] != 0]
        feature_count = len(imp_features)
        if feature_count> 100:
            iter = list(range(10, 100, 10)) + list(range(100, feature_count, 50)) + [feature_count]
        else:
            iter = list(range(10, feature_count, 10)) + [feature_count]
        target = df[y[0]]
        for i in iter:
            print('feature count {0}'.format(i))
            logging.debug("feature count{0}".format(i))
            curr_features = imp_features.feature_names[:i]
            curr_X = df.loc[:,curr_features]
            model = XGBClassifier(seed = 10)
            model.fit(curr_X, target)
            joblib.dump(model, self.dictionary['path']+'/'+'saved_objects/xgb_' + str(i) + '_features_'+ version + '.joblib', compress = 1)
            if save_pred:
                pred = model.predict_proba(df.drop(drop_vars, axis=1))
                out['pred'] = pred[:,1]
                out.to_csv(self.dictionary['path']+'/'+'results/pred_dev_'+ str(i) + '_features_'+ version + '.csv', index=False)
            feature_imp = pd.DataFrame({'feature_names': curr_features, 'importance': model.feature_importances_})
            feature_imp.to_csv(self.dictionary['path']+'/'+'results/feature_importance_' + str(i) + '_features_'+ version + '.csv', index=False)
            score = model.predict_proba(curr_X)
            logging.debug("score is : {}".format(score))
            ks = self.ksTable(score[:,1], target, 'dev_xgb_' + str(i) + '_features_' + version)
            logging.debug("Ks is : {}".format(ks))
            breaks = np.diff(ks['No.Res']) > 0
            dec_break = (np.diff(ks['No.Res']) > 0).any()
            ks_val = ks.KS.max()
            ks_decile = ks.KS.idxmax() + 1
            capture = ks['percent_cum_res'][3]
            if dec_break:
                break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
                summary_df = summary_df.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture]],columns=['feature_count', 'dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
            else:
                break_dec = np.nan
                summary_df = summary_df.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture]],columns=['feature_count', 'dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
        summary_df.to_csv(self.dictionary['path']+'/'+'results/summary_df_features_xgb_' + version + '.csv', index =False)
        logging.debug("saveFeatureIterationDev executed successfully. Dictionary is : {}".format(self.dictionary))
        
    def saveFeatureIterationsVal(self,df,dset,save_pred = False):
        logging.debug("inside saveFeatureIterationsVal module of Feature Engineering.")
        logging.debug("Iterating on different feature combination in {}..".format(dset))
        print('Iterating on different feature combination in {} ..'.format(dset))
        id_var=self.dictionary['id']
        time_var=self.dictionary['performance']
        y=self.dictionary['target']
        version=self.version
        out = df.loc[:, id_var + time_var]
        out['actual'] = df[y[0]]
        summary_df_test =  pd.DataFrame()
        print('in test feature iter function ..')
        logging.debug("in test feature iter function...")
        imp_features_df = pd.read_csv(self.dictionary['path']+'/'+'results/feature_importance_' + version + '.csv')
        imp_features_df = imp_features_df.sort_values('importance', ascending = False)
        imp_features = imp_features_df[imp_features_df['importance'] != 0]
        feature_count = len(imp_features)
        if feature_count> 100:
            iter = list(range(10, 100, 10)) + list(range(100, feature_count, 50)) + [feature_count]
        else:
            iter = list(range(10, feature_count, 10)) + [feature_count]
        target = df[y[0]]
        summary_df = pd.read_csv(self.dictionary['path']+'/'+'results/summary_df_features_xgb_' + version + '.csv')
        for i in iter:
            print(dset + ' iteration {}'.format(i))
            logging.debug(dset+" iteration{}".format(i))
            curr_features = imp_features.feature_names[:i]
            curr_X = df.loc[:,curr_features]
            model = joblib.load(self.dictionary['path']+'/'+'saved_objects/xgb_' +  str(i) + '_features_'+ version + '.joblib')
            if save_pred:
                pred = model.predict_proba(curr_X)
                out['pred'] = pred[:,1]
                out.to_csv(self.dictionary['path']+'/'+'results/pred_' + dset + '_' + str(i) + '_features_'+ version + '.csv', index=False)
            score = model.predict_proba(curr_X)
            logging.debug("score is : {}".format(score))
            ks = self.ksTable(score[:,1], target, dset +  '_xgb_' + str(i) + '_features_' + version)
            logging.debug("Ks is : {}".format(ks))
            breaks = np.diff(ks['No.Res']) > 0
            dec_break = (np.diff(ks['No.Res']) > 0).any()
            ks_val = ks.KS.max()
            ks_decile = ks.KS.idxmax() + 1
            capture = ks['percent_cum_res'][3]
            if dec_break:
                break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
                summary_df_test = summary_df_test.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
            else:
                break_dec = np.nan
                summary_df_test = summary_df_test.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
        summary_df_test.reset_index(drop=True, inplace =True)
        summary_df[dset + '_ks'] = summary_df_test[dset + '_ks']
        summary_df[dset +'_ro_break'] = summary_df_test[dset +'_ro_break']
        summary_df[dset + '_ks_decile'] = summary_df_test[dset + '_ks_decile']
        summary_df[dset + '_capture'] = summary_df_test[dset + '_capture']
        summary_df['dev_' + dset + '_ks_diff'] = (summary_df['dev_ks'] - summary_df[dset + '_ks'])*100/summary_df['dev_ks']
        summary_df.to_csv(self.dictionary['path']+'/'+'results/summary_df_features_xgb_' + version + '.csv', index =False)
        logging.debug("saveFeatureIterationVal executed successfully. Dictionary is : {}".format(self.dictionary))
    
    
    def ksTable(self,score, response, identifier):
        logging.debug("inside KS Table Module of Feature Engineering Class. score={} , response={} , identifier={}".format(score,response,identifier))
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
        logging.debug("ksTable Module executed successfully. Dictionary is : {}".format(self.dictionary))
        return(agg)

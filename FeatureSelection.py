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

class FeatureSelection:
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
        
    def featuresIterationsSummary(self):
        logging.debug("inside featureIterationsSummary Module of Feature Engineering Class.")
        iter_type = 'features'
        identifier = 'xgb_'+self.version
        summary_df = pd.read_csv('results/summary_df_' + iter_type + '_' + identifier + '.csv')
        '''
        if top > summary_df.shape[0]:
            print('Top {} iterations with features are :'.format(top))
            print(summary_df.feature_count)
        '''
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
        summary_df.to_csv('results/summary_df_' + iter_type + '_' + identifier + '_ordered.csv', index=False)
        logging.debug("featuresItearationsSummary Module executed Successfully. dictionary is : {}".format(self.dictionary))
        return summary_df



    def featuresSelection(self,rank):
        logging.debug("inside feature Selection Module of Feature Engineering Class. User choosen rank value is : {}".format(rank))
        version=self.version
        summary_df=pd.read_csv('results/summary_df_features_xgb_' + version + '_ordered.csv')
        count=int(summary_df[summary_df.Rank2== rank].iloc[0]['feature_count'])
        importance_df=pd.read_csv('results/feature_importance_'+str(count)+'_features_'+version+'.csv')
        features=list(importance_df.iloc[:,0])
        logging.debug("List of Selected Features : {}. \n Dictionary is : {}".format(features,self.dictionary))
        return features

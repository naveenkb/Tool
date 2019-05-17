import logging
logging.captureWarnings(True)
import pandas as pd
import numpy as np
import logging
import os
import pickle
import shutil

class main:
    def __init__(self):
        self.objects={}
        self.version=None
        self.ismymodel=False
        self.isruninitializer=False
        self.ismissingfit=False
        self.isdummyfit=False
        self.isoutlier=False
        self.ismissingtransform=False
        self.isdummytransform=False
        self.isoutliertransform=False
        self.isfeaturefit=False
        self.isfeaturetransform=False
        self.isfeatureselection=False
        self.isfeatureiteration=False
        self.isfeaturesselection=False
        self.ismodelselection=False
        self.istraintestsave=False
        self.isxgbparamsiterationssummary=False
        self.isxgbparamsselection=False
        self.ispdpvarreduction=False
        self.isfinalreport=False
        
        
        
    def checkfile(self,filename):
        return os.path.exists(str(str(os.getcwd())+str('/')+str('save.p')))
        
    def modelDev(self,path,test_size=0.2,version=None):
        #if(os.path.exists('save.p')==True):
            #os.chdir(os.getcwd())
            #self=pickle.load(open('save.p', 'rb'))
            
            
        try:
            os.chdir(os.getcwd())
            from MVP_Logger1.dataDictionary import myModel
            if(self.ismymodel==False):
                self.objects[version] = myModel(path)
                self.ismymodel=True
                
                
                
                
            self.objects[version].version=version
            
            '''
            Initialization
            '''
            if(self.isruninitializer==False):
                self.objects[version].train,self.objects[version].itv,self.objects[version].otv,self.objects[version].dictionary=self.objects[version].runInitializer(test_size=test_size)
                self.isruninitializer=True
                
            if (self.objects[version].otv is None):
                print("OTV dataset is not provided, ITV dataset will be considered as OTV data for Model Building")
                self.objects[version].otv=self.objects[version].itv
            
            '''
            Missing value fit and transform
            '''
            if(self.ismissingfit==False):
                self.objects[version].missingFit(self.objects[version].train, version, self.objects[version].dictionary)
                self.ismissingfit=True
            
            if(self.ismissingtransform==False):
                self.objects[version].impute_train=self.objects[version].missingTransform(self.objects[version].train)
                self.objects[version].impute_itv=self.objects[version].missingTransform(self.objects[version].itv)
                self.objects[version].impute_otv=self.objects[version].missingTransform(self.objects[version].otv)
                self.ismissingtransform=True
            
            '''
            categorical column dummy fit and transform
            '''
            if(self.isdummyfit==False):
                self.objects[version].dummiesFit(self.objects[version].impute_train, self.objects[version].dictionary, max_cat_levels = 50, na_dummies = True, version=self.objects[version].version)
                self.isdummyfit=True
            
            if(self.isdummytransform==False):
                self.objects[version].impute_train=self.objects[version].dummyTranform(self.objects[version].impute_train)
                self.objects[version].impute_itv=self.objects[version].dummyTranform(self.objects[version].impute_itv)
                self.objects[version].impute_otv=self.objects[version].dummyTranform(self.objects[version].impute_otv)
                self.isdummytransform=True
            
            '''
            Outlier fit and transform
            '''
            if(self.isoutlier==False):    
                self.objects[version].outlierFit(self.objects[version].impute_train, self.objects[version].dictionary, outlier_cap = 0.99, version=self.objects[version].version)
                self.isoutlier=True
            
            if(self.isoutliertransform==False):
                self.objects[version].impute_train=self.objects[version].outlierTransform(self.objects[version].impute_train)
                self.objects[version].impute_itv=self.objects[version].outlierTransform(self.objects[version].impute_itv)
                self.objects[version].impute_otv=self.objects[version].outlierTransform(self.objects[version].impute_otv)
                self.isoutliertransform=True
            
            #self.objects[version].merged_train=self.objects[version].impute_train
            #self.objects[version].merged_itv=self.objects[version].impute_itv
            #self.objects[version].merged_otv=self.objects[version].impute_otv
            
            '''
            Feature engineering for wide and long format
            '''
            if self.objects[version].dictionary['data_struct']=="widef":
                if(self.isfeaturefit==False):
                    self.objects[version].merged_train=self.objects[version].featureFitWide(self.objects[version].impute_train,self.objects[version].dictionary,self.objects[version].version,"dev")
                    self.isfeaturefit=True
                if(self.isfeaturetransform==False):
                    self.objects[version].merged_itv=self.objects[version].featureTransformWide(self.objects[version].impute_itv,self.objects[version].version,"itv")
                    self.objects[version].merged_otv=self.objects[version].featureTransformWide(self.objects[version].impute_otv,self.objects[version].version,"otv")
                    self.isfeaturetransform=True
                    
            elif self.objects[version].dictionary['data_struct']=="longf":
                if(self.isfeaturefit==False):
                    self.objects[version].merged_train=self.objects[version].featureFit(self.objects[version].impute_train,desc_dict=self.objects[version].dictionary,version=self.objects[version].version,cohort_period_type="M",feature_type=['c','s','v'],centrality_period=3,centrality_order=4,n_month=12,split_type="dev")
                    self.isfeaturefit=True
                if(self.isfeaturetransform==False):
                    self.objects[version].merged_itv=self.objects[version].featureTransform(self.objects[version].impute_itv,self.objects[version].version,"itv")
                    self.objects[version].merged_otv=self.objects[version].featureTransform(self.objects[version].impute_otv,self.objects[version].version,"otv")
                    self.isfeaturetransform=True
            
            '''
            Feature selection iterations
            '''
            if(self.isfeatureselection==False):
                self.objects[version].featureSelection(self.objects[version].dictionary,self.objects[version].version)
                self.isfeatureselection=True
            
            if(self.isfeatureiteration==False):
                self.objects[version].summary_df=self.objects[version].featureIterationSummary(version=self.objects[version].version)
                self.isfeatureiteration=True
                
            ##Rank has to be asked from the user
            
            '''
            Model selection from feature iteration
            '''
            if(self.isfeaturesselection==False):
                self.objects[version].featureslist=self.objects[version].featuresSelection(rank=1,version=self.objects[version].version) 
                self.isfeaturesselection=True
            if(self.ismodelselection==False):    
                self.objects[version].modelSelection(self.objects[version].dictionary,self.objects[version].version,"dev")
                self.ismodelselection=True
            
            '''
            Performs parameter iteration
            '''
            if(self.istraintestsave==False):    
                self.objects[version].train_test_save(self.objects[version].merged_train,self.objects[version].featureslist,self.objects[version].version,"dev")
                self.objects[version].train_test_save(self.objects[version].merged_itv,self.objects[version].featureslist,self.objects[version].version,'itv')
                self.objects[version].train_test_save(self.objects[version].merged_otv,self.objects[version].featureslist,self.objects[version].version,'otv')
                self.istraintestsave=True
            
            '''
            parameter summary
            '''
            if(self.isxgbparamsiterationssummary==False):
                self.objects[version].params_iterations_summary=self.objects[version].XgbParamsIterationsSummary(self.objects[version].version)
                self.isxgbparamsiterationssummary=True
            
            '''
            Model selection from parameter iteration
            '''
            if(self.isxgbparamsselection==False):
                self.objects[version].final_params= self.objects[version].XgbParamsSelection(1,self.objects[version].version)
                self.isxgbparamsselection=True
                
            '''
            Variable reduction by partial dependency plots
            '''    
            if(self.ispdpvarreduction==False):    
                self.objects[version].final_nonflat_features=self.objects[version].pdpVarReduction(self.objects[version].merged_train,[self.objects[version].merged_itv,self.objects[version].merged_otv],['itv','otv'],self.objects[version].featureslist, self.objects[version].final_params,self.objects[version].version)
                self.ispdpvarreduction=True
            
            '''
            Final report generation
            '''
            if(self.isfinalreport==False):
                self.objects[version].finalReport(self.objects[version].final_params,self.objects[version].final_nonflat_features,self.objects[version].impute_otv,self.objects[version].version)
                self.isfinalreport=True
        except KeyboardInterrupt:
            print("object saving ...")
            os.chdir(os.getcwd())
            if(os.path.exists(str(str(os.getcwd())+str('/')+str('save.p')))):
                shutil.os.remove('save.p')
            pickle.dump(self,open('save.p','wb'))
            print("object saved successfully Keyboard interrupt")
        #except IOError:
            #print("object saved successfully IO Error")
            #pickle.dump(self,open('save.p','wb'))
    
    def scoring(self,otv_name,version=None):
        otv=self.objects[version].read_data(otv_name)
        self.objects[version].impute_otv=self.objects[version].missingTransform(otv)
        self.objects[version].merged_otv=self.objects[version].featureTransform(self.objects[version].impute_otv,self.objects[version].version,"otv")
        self.objects[version].train_test_save(self.objects[version].impute_otv,self.objects[version].featureslist,self.objects[version].version,'otv')
        #self.objects[version].final_nonflat_features=self.objects[version].obj.pdpVarReduction(self.objects[version].merged_train,[self.objects[version].merged_itv,self.objects[version].merged_otv],['itv','otv'],self.objects[version].featureslist, self.objects[version].final_params,self.objects[version].version)
        self.objects[version].finalReport(self.objects[version].final_params,self.objects[version].final_nonflat_features,self.objects[version].impute_otv,self.objects[version].version)
    
    def cleaning(self):
        shutil.os.remove('save.p')
    
    def save(self):
        if(os.path.exists(str(str(os.getcwd())+str('/')+str('save.p')))):
            shutil.os.remove('save.p')
        pickle.dump(self,open('save.p','wb'))
        #shutil.os.remove('model.log')
    

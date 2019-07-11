"""
Version v0.9.2

"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import math
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals import joblib
import pandas as pd
import datetime
import dateutil.relativedelta
import numpy as np
import logging
from collections import defaultdict
from sklearn.externals import joblib
import pickle

class myModel:
    """
    Parameters  : 
        input a path while creating an object and code will output all input files and take output files from this path only.
        self.basic_dict['data_dict_name'] = "data_dictionary.csv"  - a generalised name , this name will be same through out the process
        self.basic_dict['performace_format']=list()   - creating a list to get the format of PERFORMANCE Variable for date standardisation. 
        self.basic_dict['date_format']=list()         - an empty list this will store format for other dates if available in data ( e.g. - order_creation_Date etc.)
        self.basic_dict['path']='/'.join(path.split('/'))  - spliting a path based on / and joining again to remove any extra /
        os.chdir(path)   - changing current directory
        self.max_category=50 -  setting up default max category for the categorical variable. user can change whenever required in the subsequent function calls.
    """ 
    def __init__(self,path):
        self.basic_dict=dict()
        self.basic_dict['data_dict_name']="data_dictionary.csv"
        self.basic_dict['performace_format']=list()
        self.basic_dict['date_format']=list()
        self.basic_dict['path']='/'.join(path.split('/'))
        self.basic_dict['putInitial']=False
        self.basic_dict['putDataDictionary']=False
        os.chdir(path)
        self.max_category=50
        self.feature={}
        self.mdlSelection={}
        self.feature_sel={}
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.DEBUG,format="%(asctime)s:%(levelname)s:%(message)s",filename="model.log")
        #logger = logging.get_logger()
        #logger.warning('This should go in the file.')
        #print logger.handlers
        logging.debug("myModel Class Object Created with Dictionary :{}  , max value for categorical fields are:{}".format(self.basic_dict,self.max_category))
        

        #self.Imputer=Imputer()
    
    """ put_Initialfile is used to provide a file to modeller with some basic varibales like :
        
        path - path of directory
        dev_data_name - name of dev data with extension or without extension
        otv_data_name - name of otv data with extension or without extension - not mandatory
        itv_data_name - name of itv data with extension or without extension - not mandatory , if itv data is not available then we will use train_test_split
            dev_cohort , otv_cohort - provide only if dev and otv data file is one, mandatory if otv data and train data both are in the same file.
   """
    def put_InitialFile(self):
        logging.debug("Inside put_Initial Module.")
        #cur_dir=os.curdir
        columns=['path','data_name','itv_data_name','otv_data_name','dev_cohort','otv_cohort','data_struct']
        formats = ['string','string','string','string','Please Fill the date Format','Please Fill the date Format','please fill the structure of data']
        provided=[self.basic_dict['path'],"","","","","",""]
        comments=['Path to your input data files','Name of developement data with .csv format e.g test_data.csv','Name of ITV data. do not fill if you dont have ITV data','Name of OTV data. do not fill if you dont have OTV data separate','developement cohort format same as performace period format in data. like - YYYYMM for 201802','OTV cohort format same as performace period format in data. like YYYYMM for 201803','Please fill structure of data. if data is in wide format then fill widef . if data is in long format please fill longf']
        basic_data=pd.DataFrame({'Requirements':columns,'user_provided_details':provided,'user_provided_format':formats,'comments':comments})
        basic_data.to_csv(self.basic_dict['path']+'/basic_data_requiremnts.csv',columns=['Requirements','user_provided_details','user_provided_format','comments'],index=False)
        self.basic_dict['putInitial']=True
        logging.debug("basic_data_requirements.csv file is placed at path:{}".format(self.basic_dict['path']))
        
        '''
        Read the Basic Requirement file and create Dynamic variables for all the basic fileds which are available in basic_requirements file.
       '''     
        
    def get_InitialFile(self):
        logging.debug("Inside get_Initial Module.")
        basic_data=pd.read_csv(self.basic_dict['path']+'/basic_data_requiremnts.csv')
        basic_data = basic_data.where((pd.notnull(basic_data)), None)
        variable_name=basic_data.loc[:,'Requirements']
        variable_val=basic_data.loc[:,'user_provided_details']
        variable_formats=basic_data.loc[:,'user_provided_format']
        for i in range(len(variable_name)):
            self.basic_dict[variable_name[i]]=variable_val[i]
            if(variable_formats[i]!="string"):
                self.basic_dict[variable_name[i]+"_format"]=variable_formats[i]
        
        if (self.basic_dict['data_struct']=="widef" and (self.basic_dict['itv_data_name']==None or self.basic_dict['otv_data_name']==None)):
            logging.debug("can not go ahead without OTV and ITV data sets.")
            raise Exception("can not go ahead without OTV and ITV data sets.")
                
        if (self.basic_dict['data_struct']=="longf" and (self.basic_dict['dev_cohort']==None or self.basic_dict['otv_cohort']==None or self.basic_dict['data_name']==None)):
            raise Exception("Following fields of basic_requirement.csv --> dev_cohort , otv_cohort , data_name CAN NOT BE BLANK")
            
        cohorts_df=pd.DataFrame(data=[self.basic_dict['dev_cohort'] ,self.basic_dict['otv_cohort']],columns=['cohort'],index=['dev','otv'])
        if self.basic_dict['data_struct']=="longf":
            ch_fmt_dev=self.basic_dict['dev_cohort_format']
            ch_fmt_otv=self.basic_dict['otv_cohort_format']
            if ch_fmt_dev==ch_fmt_otv:
                cohorts_df['cohort']= pd.to_datetime(cohorts_df['cohort'],errors='raise',format=ch_fmt_dev)
            else:
                raise Exception('Format of dev and otv cohorts needs to be same')
                
        self.basic_dict["cohort_df"]=cohorts_df
        logging.debug("A File read with name basic_data_requirements.csv from path : {}. Dictionary has the values:{}".format(self.basic_dict['path'] , self.basic_dict))
        self.check_Path()

        
    '''
    Check for the Path accesses, User must have read and write access for the Directory. 
    if User Dont have access for the Directory then we will not be able to process Further
    '''
    def check_Path(self,new_path=None):
        logging.debug("Inside check_path Module.")
        if(new_path!=None):
            self.basic_dict['path']=new_path
            
        self.basic_dict['path']='/'.join(self.basic_dict['path'].split('/'))
        #self.path="/"+self.path
        try:
            read_f=0
            write_f=0
            if(os.access(self.basic_dict['path'], os.R_OK)):
                read_f=1
            if(os.access(self.basic_dict['path'], os.W_OK)):
                write_f=1
            if(not read_f and not write_f):
                print("Directory does not exist or You dont have (Read and write) access for the given ",self.basic_dict['path'])
                logging.debug("Directory does not exist or You dont have (Read and write) access for the given path:{}".format(self.basic_dict['path']))
                return False
            
            logging.debug("path : {} Exist for Reading and Writing".format(self.basic_dict['path']))
            self.change_Path()
        except:
            print("Not a valid selection check_pathaccess")
            return False
  

    '''
     in this part i am changing the current Directory to path provided by User.
     
    '''  
    def change_Path(self,new_path=None):
        logging.debug("Inside change_path Module.")
        if(new_path!=None):
            self.check_Path(new_path)    
        logging.debug("path: {} changed to given path in Dictionary :{}".format(new_path , self.basic_dict['path']))
        self.check_File()
            
            
            
    '''
    Checking for the File. if dev_data File Exist then only we will process further.
    
    '''

    def check_File(self):
        logging.debug("Inside Check File Module.")
        #os.chdir(self.basic_dict['path'])
        try:
            if(self.basic_dict['data_name']==None):
                print("name of data can not be null. Sorry i am unable to process further")
                logging.debug("name of data can not be null. Sorry i am unable to process further")
                return False
        
            self.basic_dict['data_name']=str(self.basic_dict['data_name'].split(".csv")[0])+str(".csv")
            #print(str(str(self.basic_dict['path'])+str('/data/')+str(self.basic_dict['data_name'])))
        
            if(os.path.exists(str(str(self.basic_dict['path'])+str('/')+str(self.basic_dict['data_name'])))):
                if(self.basic_dict['data_name']!=None):
                    self.dev_data=pd.read_csv(self.basic_dict['path']+"/"+self.basic_dict['data_name'])
                    logging.debug("File Exist {} at path {} and Successfully read and stores in the pandas dataframe name dev_data. Dictionary is : {}".format(self.basic_dict['dev_data_name'], self.basic_dict['path'] ,  self.basic_dict))
                else:
                    raise Exception("data name is not provided correctly")
                return True
            else:
                #print("File does not exist, please check and provide the correct name of file")
                logging.debug("File does not exist, please check and provide the correct name of file")
                return False
        except:
            logging.debug("Not a valid selection in check_File")
            return False
        return True 
        
    
    ###### Reading All the Dataset given / name provided by user in the basic requiremnts.csv
    
    def read_Data(self,filename=None,path=None):
        logging.debug("Inside read_data Module. filename : {}  and path : {} . Dictionary is : {}".format(filename , path , self.basic_dict))
        if(path==None):
            path=self.basic_dict['path']
        if(filename!=None):
            filename=str(filename.split(".csv")[0])+str(".csv")
            return self.StandardizeDatatype(pd.read_csv(path+"/"+filename))
        else:
            raise Exception("File name does not exist or file name is not correct")
        
        
    """
    #Generate a Data Dictionary for User with Default Variable category. user can edit and save the file. 
    
    #ID Variable - id
    #Performance / cohort variable - per
    #sequence variable - seq
    #categorical variable  - cat
    #numerical variable - num
    #other time variables like order date etc - date
    #target variable - target
    """
    def put_DataDictionaryWide(self,max_category=None):
        logging.debug("Inside put_DataDictionaryWide Module.")
        if(max_category!=None):
            self.max_category=max_category
        
        col_names=self.dev_data.columns
        col_dict=defaultdict(list)
        for i in col_names:
            lst=i.split('-')
            if(lst[-1:][0].isdigit()):
                lst1=lst[:-1]
                var='-'.join(lst1)
                #print(var)
                col_dict[var].append(int(lst[-1:][0]))
            else:
                lst='-'.join(lst)
                col_dict[lst]=lst    
      
        var=[]
        mini=[]
        maxim=[]
        dtype=[]
        for key,value in col_dict.items():
            if(isinstance(value,list)):
                var.append(key)
                mini.append(min(value))
                maxim.append(max(value))
                dtype.append("seq")
            else:
                var.append(key)
                mini.append(0)
                maxim.append(0)
                dtype.append(["num" if self.dev_data[key].dtypes in ["float64"] else 'cat' if self.dev_data[key].dtypes in ["O"]  else "cat" if self.dev_data[key].dtypes in ["int64"] and len(list(self.dev_data[key].unique())) <= self.max_category else "cat" if self.dev_data[key].dtypes in ["O"] and len(list(self.dev_data[key].unique())) <self.max_category else "num"][0])
                        
        output=pd.DataFrame({"col":var , "start_index":mini , "end_index":maxim , "suggested_dtype":dtype,"user_edited_dtype":dtype},columns=["col","start_index","end_index","suggested_dtype","user_edited_dtype"])
        self.data_dict=None
        output.to_csv(self.basic_dict['path']+'/'+'data_dictionary.csv',index=False)
        self.basic_dict['putDataDictionary']=True
        logging.debug("Put Data Dictionary module executed Successfully. Dictionary is : {}".format(self.basic_dict))
     


    def get_DataDictionaryWide(self,test_size=0.2):
        logging.debug("Inside get_DataDictionaryWide Module with test_size : {}".format(test_size))
        self.data_dict=pd.read_csv(self.basic_dict['path']+'/'+self.basic_dict['data_dict_name'])
        col=self.data_dict['col']
        start=self.data_dict['start_index']
        end=self.data_dict['end_index']
        dtype=self.data_dict['user_edited_dtype']
        self.basic_dict['performance']=[]
        self.basic_dict['id']=[]
        self.basic_dict['seq']=[]
        self.basic_dict['cat']=[]
        self.basic_dict['drop']=[]
        self.basic_dict['target']=[]
        self.basic_dict['num']=[]
        
        for i in range(len(col)):
            if dtype[i][0]=="s":
                j=start[i]
                for j in range(start[i],end[i]):
                    self.basic_dict['seq'].append(col[i]+"-"+str(j))
            
            elif(dtype[i][0]=="p"):
                self.basic_dict['performance'].append(col[i])
            elif(dtype[i][0]=="n"):
                if start[i]==0 and end[i]==0:
                    self.basic_dict['num'].append(col[i])
                elif(start[i] < end[i]):
                    j=start[i]
                    for j in range(start[i],end[i]):
                        self.basic_dict['num'].append(col[i]+"-"+str(j))
            elif(dtype[i][0]=="c"):
                if start[i]==0 and end[i]==0:
                    self.basic_dict['cat'].append(col[i])
                elif(start[i] < end[i]):
                    j=start[i]
                    for j in range(start[i],end[i]):
                        self.basic_dict['cat'].append(col[i]+"-"+str(j))
            elif(dtype[i][0]=="i"):
                self.basic_dict['id'].append(col[i])
            
            elif(dtype[i][0]=="t"):
                self.basic_dict['target'].append(col[i])
            elif(dtype[i][0]=="d"):
                j=start[i]
                k=end[i]
                if ((k-j)!=0):
                    for j in range(start[i],end[i]):
                        self.basic_dict['drop'].append(col[i]+"-"+str(j))
                else:
                    self.basic_dict['drop'].append(col[i])
                    #self.basic_dict['drop'].append(col[i])
        self.basic_dict['num'].extend(self.basic_dict['seq'])            
        self.basic_dict['seqvaronly']=np.where(self.data_dict['user_edited_dtype']=='seq',self.data_dict.col,99999)
        self.basic_dict['seqvarstart']=np.where(self.data_dict['user_edited_dtype']=='seq',self.data_dict['start_index'],99999)
        self.basic_dict['seqvarend']=np.where(self.data_dict['user_edited_dtype']=='seq',self.data_dict['end_index'],99999)
        logging.debug("Data Dictionary read and created successfully.")
        logging.debug("ID Variables from Data : {}".format(self.basic_dict['id']))
        logging.debug("performace Variables from Data : {}".format(self.basic_dict['performance']))
        logging.debug("Sequence Variables from Data : {}".format(self.basic_dict['seq']))
        logging.debug("Categorical Variables from Data : {}".format(self.basic_dict['cat']))
        #logging.debug("date Variable from Data : {}".format(self.basic_dict['date']))
        otv_data=None
        
        self.dev_data=self.StandardizeDatatype(self.dev_data)
        if(self.basic_dict['otv_data_name']!=None):
            otv_data=self.StandardizeDatatype(self.read_Data(self.basic_dict['otv_data_name']))
            
        elif(self.basic_dict['cohort_df']['cohort']['otv']!=None):
            self.dev_data,otv_data=self.splitDevOtvData(self.dev_data)
    
        
        if self.basic_dict['itv_data_name']==None:
            logging.debug("itv data was not given in basic requirement so that's why train , Itv split with test_size: {}".format(test_size))
            #print("in if")
            Train,itv=self.splitDevItvData(self.dev_data,test_size)
            logging.debug("Returning dev , itv and Data dictionary from get_DataDictionary Module")
            return Train,itv,otv_data,self.basic_dict
        else:
            logging.debug("reading given ITV data in Basic Requirements file : {}".format(self.basic_dict['itv_data_name']))
            logging.debug("Returning dev , itv and Data dictionary from get_DataDictionary Module")
            #print("in else")
            return self.dev_data,self.StandardizeDatatype(self.read_Data(self.basic_dict['itv_data_name'])),otv_data,self.basic_dict
        
            
    def put_DataDictionary(self,max_category=None):
        logging.debug("Inside put_DataDictionary Module.")
        if(max_category!=None):
            self.max_category=max_category
        var=self.dev_data.columns
        var_type=self.dev_data.dtypes
        var_type_format="Fill the format only if its a date or time variable."
        suggested_type=["num" if var_type[x] in ["float64"] 
                                 else "cat" if ((var_type[x] in ["int64"]) and len(list(self.dev_data[var[x]].unique()))<=self.max_category)
                                    else "cat" if ((var_type[x] in ["O"]))
                                        else "date" if isinstance(self.dev_data[var[x]],datetime.date) 
                                            else "num" for x in range(len(var)) ] 
        self.data_dict=None
        pd.DataFrame({"Variable":self.dev_data.columns,"suggested_type":suggested_type,"user_edited_type":suggested_type,"user_edited_format":var_type_format}).to_csv(self.basic_dict['path']+'/'+'data_dictionary.csv',columns=["Variable","suggested_type","user_edited_type","user_edited_format"],index=False)
        self.basic_dict['putDataDictionary']=True
        logging.debug("Put Data Dictionary module executed Successfully. Dictionary is : {}".format(self.basic_dict))
      
      
       
    '''
    Reading the Variables using category provided by modeller. 
    
    input - data dictionary
    output - list of each category
    '''
    
    def ext_vars(self,df=None,type_var="dummy"):
        logging.debug("Inside ext_vars Module.")
        try:
            var_name=df['Variable']
            var_type=df['user_edited_type']
            user_edited_format=df['user_edited_format']
            final=list()
            for i in range(len(var_name)):
                if(var_type[i][0]==type_var[0]):
                    final.append(var_name[i])
                if(type_var=="per" and var_type[i][0]=="p"):
                    self.basic_dict['performace_format'].append(user_edited_format[i])  
            logging.debug("List of {} Variable is :{} ".format(type_var,final))
            return final
        except:
            print(str(type_var)+"  variable tag is not identified in proper order")
            return [None]
    
    '''
    Extraction of only numeric variables.
    input - data dictionary  , time variables , id variables , sequence variables , categorical variables , date variables , target variables.
    out put - list for each type of variables.
    
   '''
   
   
    def ext_other_var(self,df=None,time_var=[None],id_vars=[None],seq_vars=[None],cat_vars=[None],drop_vars=[None],target=[None]):
        logging.debug("Inside ext_other_var Module.")
        lcols=df.Variable
        b_list=[x for x in time_var]
        b_list.extend(id_vars)
        b_list.extend(target)
        b_list.extend(seq_vars)
        b_list.extend(cat_vars)
        b_list.extend(drop_vars)
        l=[x for x in lcols if x not in b_list]
        logging.debug("List of those Which variable which do not come under ID , categorical , sequence , drop and target. but come under numerical. List is : {}".format(l))
        return list(l)
       
       
       
        '''
            Reading the Data Dictionary Provided by us and edited by User.     
        '''
        
    def get_DataDictionary(self,test_size=0.2):
        logging.debug("Inside get_DataDictionary Module with test_size : {}".format(test_size))
        self.data_dict=pd.read_csv(self.basic_dict['path']+'/'+self.basic_dict['data_dict_name'])
        self.data_dict['user_edited_type'] = self.data_dict.apply(lambda row: 'num' if row['user_edited_type'][0]=='n' else ('seq'  if row['user_edited_type'][0]=='s' else ('per'  if row['user_edited_type'][0]=='p' else ('drop'  if row['user_edited_type'][0]=='d' else ('cat'  if row['user_edited_type'][0]=='c' else ('target'  if row['user_edited_type'][0]=='t' else ('id' if row['user_edited_type'][0]=='i' else "")))))),axis=1) 
        self.data_dict['user_edited_type']=self.data_dict['user_edited_type'].str.lower()
        self.basic_dict['performance']  = self.ext_vars(self.data_dict,type_var="per")
        self.basic_dict['id']    = self.ext_vars(self.data_dict,type_var="id")
        self.basic_dict['seq']   = self.ext_vars(self.data_dict,type_var="seq")
        self.basic_dict['cat']   = self.ext_vars(self.data_dict,type_var="cat")
        #self.basic_dict['date']  = self.ext_vars(self.data_dict,type_var="date")
        self.basic_dict['target']= self.ext_vars(self.data_dict,type_var="target")
        self.basic_dict['drop']= self.ext_vars(self.data_dict,type_var="drop")
        self.basic_dict['other'] = self.ext_other_var(self.data_dict,self.basic_dict['performance'] , self.basic_dict['id'] , self.basic_dict['seq'], self.basic_dict['cat'] , self.basic_dict['drop'] , self.basic_dict['target'])
        num_var=[x for x in self.basic_dict['seq']]
        num_var.extend(self.basic_dict['other'])
        self.basic_dict['num']=num_var
        #self.data_dict.to_csv(self.basic_dict['path']+'/'+'data_dictionary.csv',index=False)
        logging.debug("Data Dictionary readed and created successfully.")
        logging.debug("ID Variables from Data : {}".format(self.basic_dict['id']))
        logging.debug("performace Variables from Data : {}".format(self.basic_dict['performance']))
        logging.debug("Sequence Variables from Data : {}".format(self.basic_dict['seq']))
        logging.debug("Categorical Variables from Data : {}".format(self.basic_dict['cat']))
        #logging.debug("date Variable from Data : {}".format(self.basic_dict['date']))
        otv_data=None
        
        self.dev_data=self.StandardizeDatatype(self.dev_data)
        if(self.basic_dict['otv_data_name']!=None):
            otv_data=self.StandardizeDatatype(self.read_Data(self.basic_dict['otv_data_name']))
            
        elif(self.basic_dict['cohort_df']['cohort']['otv']!=None):
            self.dev_data,otv_data=self.splitDevOtvData(self.dev_data)
    
        
        if self.basic_dict['itv_data_name']==None:
            logging.debug("itv data was not given in basic requirement so that's why train , Itv split with test_size: {}".format(test_size))
            #print("in if")
            Train,itv=self.splitDevItvData(self.dev_data,test_size)
            logging.debug("Returning dev , itv and Data dictionary from get_DataDictionary Module")
            return Train,itv,otv_data,self.basic_dict
        else:
            logging.debug("reading given ITV data in Basic Requirements file : {}".format(self.basic_dict['itv_data_name']))
            logging.debug("Returning dev , itv and Data dictionary from get_DataDictionary Module")
            #itv=self.dev_data=self.StandardizeDatatype(self.dev_data)
            return self.dev_data,self.StandardizeDatatype(self.read_Data(self.basic_dict['itv_data_name'])),otv_data,self.basic_dict
            
        
    
    
    '''
    Standardising Data Types of All Variables and All Data Types
    input - Split type of Data
    ID- object
    Time variables- pd.datetime
    Categorical - object
    Num - numeric(int or float)
    seq - numeric(int or float)
    Target- numericnumeric(int or float)
     
    
    '''
    
    def StandardizeDatatype(self,data=pd.DataFrame()):
        logging.debug("inside StandardizeDatatype Module")
        if(data.empty):
            raise Exception('Pass data_split_type as one of these three: (dev/itv/otv)!') 
            return
            
        id_list=self.basic_dict['id']
        num_seq_list=self.basic_dict['num']
        target_list=self.basic_dict['target']
        cat_list=self.basic_dict['cat']
        per_list=self.basic_dict['performance']
        drop_list=self.basic_dict['drop']
        if data.empty:
            raise Exception('CSV Dataset of the specified name is Not present in the location!')
        else:
            try:
                data[id_list] = data[id_list].astype(object,errors='ignore')
            except:
                print("check id list variable type given correctly or not\n")
            try:
                data[num_seq_list] = data[num_seq_list].astype(float, error='ignore')
            except:
                print("check numeric and sequence variable data type given correctly in file or not\n")
            """for var in num_seq_list:
                if var in data.columns.tolist():
                    #print(var)
                    if var not in data.select_dtypes(include=[np.number]).columns.tolist():
                        data[var] = data[var].astype(float,errors='ignore')"""
            
            #print("after first for")
            for var in target_list:
                if var in data.columns.tolist():
                    if var not in data.select_dtypes(include=[np.number]).columns.tolist():
                       data[var] = data[var].astype(float,errors='ignore')
            data[cat_list] = data[cat_list].astype(object,errors='ignore')
            data[per_list] = data[per_list].astype(str,errors='ignore')
            #data[time_list] = data[time_list].astype(str,errors='raise')
            
            for var in drop_list:
                if var in data.columns.tolist():
                    data=data.drop([var],axis=1)  
            counter=0
            
            for var in per_list:
                fmt=self.basic_dict['performace_format'][counter]
                data[var]= pd.to_datetime(data[var],errors='ignore',format=fmt)
                counter=counter+1
        return data
            
    ##########################################?????????????????????????????????
    def splitDevOtvData(self,df=None):
        logging.debug("inside splitDevOtvData Module")
        if(df is not None):
            if(self.basic_dict['otv_cohort']!=None):
                dev_clnt=df.loc[df[self.basic_dict['performance'][0]]==self.basic_dict['cohort_df']['cohort']['dev']][self.basic_dict['id']]
                dev=df.loc[df[self.basic_dict['performance'][0]]<=self.basic_dict['cohort_df']['cohort']['dev']]
                dev_df=pd.merge(dev,dev_clnt,how='inner',on=self.basic_dict['id'])
                
                otv_clnt=df.loc[df[self.basic_dict['performance'][0]]==self.basic_dict['cohort_df']['cohort']['otv']][self.basic_dict['id']]
                otv=df.loc[df[self.basic_dict['performance'][0]]<=self.basic_dict['cohort_df']['cohort']['otv']]
                otv_df=pd.merge(otv,otv_clnt,how='inner',on=self.basic_dict['id'])
                
                self.basic_dict['is_otv']=True
                return dev_df,otv_df
            
    def splitDevItvData(self,df=None,test_size=0.2):  
        logging.debug("inside splitDevItvData Module")         
        if(df is not None):
            train_df, itv_df = train_test_split(df, test_size = test_size, random_state = 12345)
            self.basic_dict['is_train']=True
            self.basic_dict['is_itv'] =True
            return train_df,itv_df
        else:
            raise Exception("Dataframe not provided at the time of funtional calling")
            
            
    def runInitializer(self,test_size=0.2):
        logging.debug("inside runInitializer Module")
        
        if self.basic_dict['putInitial']==False:
            self.put_InitialFile()
            print("\n basic_requirements.csv file is created at path :  " ,self.basic_dict['path'] , '\tPlease provide required details.' )
		 
        action=input("\n Confirm if necessary details are provided in basic_requirements.csv file and saved in the same location (Y/N) -------->    ").lower()
        while(action=='n'):
            action=input("\n Confirm if necessary details are provided in basic_requirements.csv file and saved in the same location (Y/N) -------->     ").lower()
        logging.debug("calling get_Initial Module")
        self.get_InitialFile()
        
        if self.basic_dict['putDataDictionary']==False:
            if self.basic_dict['data_struct']=="longf":
               self.put_DataDictionary()
            elif self.basic_dict['data_struct']=="widef":
               self.put_DataDictionaryWide()
            print("\n data_dictionary.csv file has been created at path :  ", self.basic_dict['path'] , " Please Provide the required details." )
		 
        action=input("\n Confirm if necessary details are provided in data_dictionary.csv file and saved in the same location (Y/N) -------->   ").lower()
        while(action=='n'):
             action=input("\n Confirm if necessary details are provided in data_dictionary.csv file and saved in the same location (Y/N) -------->   ").lower()

        if self.basic_dict['data_struct']=="longf":
            logging.debug("calling get_DataDictionary Module")
            return self.get_DataDictionary(test_size)
        elif self.basic_dict['data_struct']=="widef":
            logging.debug("inside get_DataDictionaryWide Module")
            return self.get_DataDictionaryWide(test_size)
        
    
    def missingFit(self, df,version="v1",basic_dict=None):
        logging.debug("inside missingFit Module")
        from Imputer import Imputer
        self.Imputer=Imputer()
        self.Imputer.fit(df,basic_dict,version)
    
    def missingTransform(self,df):
        logging.debug("inside missingTransform Module")
        return self.Imputer.transform(df)
    
    def dummiesFit(self,df, basic_dict, max_cat_levels = 50, na_dummies = True, version=None):
        logging.debug("inside dummiesFit Module")
        from Class_Dummy_Creation import CreateDummies
        self.createDummy=CreateDummies()
        self.createDummy.fit(df,basic_dict, max_cat_levels, na_dummies, version)
        
        
    def dummyTranform(self,df):
        logging.debug("inside dummyTransform Module")
        return self.createDummy.transform(df)
        
    def outlierFit(self,df, basic_dict, lower_cap, upper_cap, version=None):
        logging.debug("inside outlierFit Module")
        from Class_Outlier_Treatment import OutlierTreatment
        self.outlier=OutlierTreatment()
        self.outlier.fit(df, basic_dict, lower_cap, upper_cap, version)
        
        
    def outlierTransform(self,df):
        logging.debug("inside outlierTransform Module")
        return self.outlier.transform(df)
        
    
    def featureFitWide(self,df,desc_dict,version,split_type=None):
        logging.debug("inside featureFitWide Module of Data Dictionary class.")
        from FeatureEngineering import FeatureEngineering
        if split_type=="dev":
            self.feature[version]=FeatureEngineering(desc_dict,version)
            if self.basic_dict['data_struct']=='widef':
                merged_df=self.feature[version].createFeaturesWide(df,self.basic_dict['seqvaronly'],self.basic_dict['seqvarstart'],self.basic_dict['seqvarend'])
            self.feature[version].saveFeatureIterationsDev(merged_df)
            return merged_df
            
        
    
    def featureTransformWide(self,df,version="v1",split_type=None):
        logging.debug("inside featureTransformWide Module")
        if self.basic_dict['data_struct']=='widef' and split_type=="otv" or split_type=="itv":
                merged_df=self.feature[version].createFeaturesWide(df,self.basic_dict['seqvaronly'],self.basic_dict['seqvarstart'],self.basic_dict['seqvarend'])
        self.feature[version].saveFeatureIterationsVal(merged_df,split_type)
        return merged_df
            
    
    def featureFit(self,df,desc_dict,version,cohort_period_type=None,feature_type=['c','s','v'],centrality_period=None,centrality_order=None,n_month=12,split_type=None):
        logging.debug("inside featureFit Module of Data Dictionary class.")
        from FeatureEngineering import FeatureEngineering
        
        if split_type=="dev":
            self.feature[version]=FeatureEngineering(desc_dict,version)
            if self.basic_dict['data_struct']=='widef':
                merged_df=self.feature[version].createFeaturesWide(df,self.basic_dict['seqvaronly'],self.basic_dict['seqvarstart'],self.basic_dict['seqvarend'])
            elif self.basic_dict['data_struct']=='longf':
                if len(feature_type)>0:
                    self.feature[version].cohort_period_type=cohort_period_type
                    self.feature[version].feature_type=feature_type
                    self.feature[version].centrality_period=centrality_period
                    self.feature[version].centrality_order=centrality_order
                    self.feature[version].n_month=n_month
                    merged_df=self.Features(df,desc_dict,version,cohort_period_type,feature_type,centrality_period,centrality_order,n_month,split_type)
                else:
                    merged_df=df.loc[df[self.basic_dict['performance'][0]]==self.basic_dict['cohort_df']['cohort']['dev']]
            self.feature[version].saveFeatureIterationsDev(merged_df)
            return merged_df
            
            
    
    def featureTransform(self,df,version="v1",split_type=None):
        logging.debug("inside featureTransform Module")
        if len(self.feature[version].feature_type)>0:
            merged_df= self.Features(df,self.feature[version].dictionary,self.feature[version].version,self.feature[version].cohort_period_type,self.feature[version].feature_type,self.feature[version].centrality_period,self.feature[version].centrality_order,self.feature[version].n_month,split_type)
        else:
            if split_type=="otv":
                merged_df=df.loc[df[self.basic_dict['performance'][0]]==self.basic_dict['cohort_df']['cohort']['otv']]
            else:
                merged_df=df.loc[df[self.basic_dict['performance'][0]]==self.basic_dict['cohort_df']['cohort']['dev']]
        self.feature[version].saveFeatureIterationsVal(merged_df,split_type)
        return merged_df
        
        
    
    def Features(self,df,desc_dict,version,cohort_period_type=None,feature_type=['c','s','v'],centrality_period=None,centrality_order=None,n_month=12,split_type=None):
        logging.debug("inside Features Module")
        avail_df=[]
        if('c' in feature_type):
            logging.debug("calling centrality Feature creation Module")
            centf,dictionary=self.feature[version].createCentralityFeatures(df,cohort_period_type,centrality_period,centrality_order,split_type)
            avail_df.append(centf)
        
        if('s' in feature_type):
            if self.basic_dict['seq'] != []:
                logging.debug("calling Sequence Feature creation Module")
                seqf,dictionary=self.feature[version].createSequenceFeature(df,cohort_period_type,n_month,split_type)
                avail_df.append(seqf)
            
        if('v' in feature_type):
            logging.debug("calling velocity feature creation Module")
            vecf,dictionary=self.feature[version].velocity_fts(df,cohort_period_type,n_month,split_type)
            avail_df.append(vecf)
        
        logging.debug("calling  Module mergeRawAndCreateFeatures module for merging raw data with created features data sets")
        if(split_type=="dev" or split_type=="itv"):
            merged_df=self.feature[version].mergeRawAndCreatedFeatures(df.loc[df[self.basic_dict['performance'][0]]==self.basic_dict['cohort_df']['cohort']['dev']],avail_df)
        
        elif(split_type=="otv"):
            merged_df=self.feature[version].mergeRawAndCreatedFeatures(df.loc[df[self.basic_dict['performance'][0]]==self.basic_dict['cohort_df']['cohort']['otv']],avail_df)
        return merged_df
        
        
    def featureSelection(self,desc_dict,version):
        from FeatureSelection import FeatureSelection
        self.feature_sel[version]=FeatureSelection(desc_dict,version)
        
    def featureIterationSummary(self,version=None):
        if(version==None):
            print("No argument passed. version should be passed as an argument")
            return None
        return self.feature_sel[version].featuresIterationsSummary()
        #try:
            #print("Hi i am inside featureIterationSummary")
            #print(self.feature[version]);
            #print(self.feature[version].dictionary); 
        #print("Object didn't found")
    
    def featuresSelection(self,rank=None,version=None):
        if(rank==None):
            print("rank is mandatory argument for this module as you will select features based on the given rank in file.")
            return None
        if(version==None):
            print("version can't be empty")
            return None
        return self.feature_sel[version].featuresSelection(rank=rank)
        
    def modelSelection(self,dictionary=None,version=None,split_type=None):
        from XgbSelection import XgbSelection
        if(dictionary==None or version==None or split_type==None or dictionary=='' or version=='' or dictionary==''):
            print("you have to pass dictionary , version and split_type. please check which argument you have missed")
            return None
        if(split_type=="dev"):
            self.mdlSelection[version]=XgbSelection(dictionary,version)
            self.mdlSelection[version].runInitializer()
        
    def train_test_save(self,df,featureslist=None,version=None,split_type=None):
        
        if(featureslist==None or len(featureslist)==0 or split_type==None or split_type=='' or version==None or version==''):
            print("feature list and split type can not be Null.")
            return None
        if(split_type=="dev"):
            self.mdlSelection[version].saveTrainingsDev(df,featureslist)
        elif(split_type=="itv" or split_type=="otv"):
            self.mdlSelection[version].saveTestingsVal(df,featureslist,split_type)
    
    def XgbParamsIterationsSummary(self,version=None):
        return self.mdlSelection[version].XgbParamsIterationsSummary()
    
    def XgbParamsSelection(self,rank=None,version=None):
        if(rank==None or version==None):
            print("rank or version can not be empty")
            return None
        return self.mdlSelection[version].XgbParamsSelection(rank)
        
    def pdpVarReduction(self,df,dset_list,dset_list_name,featureslist, final_params,version=None):
        return self.mdlSelection[version].pdpVarReduction(df,dset_list,dset_list_name,featureslist,final_params)
    
    def finalReport(self,final_params,final_nonflat_features,otv,itv,version=None):
        self.mdlSelection[version].finalReport(final_params,final_nonflat_features,otv,itv)
        
        
            
        

# -*- coding: utf-8 -*-
import pandas as pd
import featuretools as ft
from featuretools.variable_types import Numeric
from featuretools.primitives import make_agg_primitive , make_trans_primitive
import time
tStart = time.time() #計時開始

alldata = pd.read_excel('C:\\Users\\vincentkuo\\Documents\\ARAMT_L1M.xlsx',encoding='utf-8')
alldata2 = pd.read_excel('C:\\Users\\vincentkuo\\Documents\\ARMINVOICE.xlsx',encoding='utf-8')
alldata3 = pd.read_excel('C:\\Users\\vincentkuo\\Documents\\CUSTOMER.xlsx',encoding='utf-8')
#dfAllData_1 = alldata.iloc[:,100:] #Get column from the 1001 column
dfAllData_1 = alldata.iloc[:,:]
dfAllData_2 = alldata2.iloc[:,:]
dfAllData_3 = alldata3.iloc[:,:]
#print(alldata)


# An EntitySet is a collection of entities and the relationships between them. 
# They are useful for preparing raw, structured datasets for feature engineering.
es = ft.EntitySet(id="test_data") 

es = es.entity_from_dataframe(entity_id="ARAMT_L1M",
                               dataframe=dfAllData_1,
                               index="CUSTOMERMCODE"
                               )

es = es.entity_from_dataframe(entity_id="ARMINVOICE",
                               dataframe=dfAllData_2,
                               index="INDEX"
                               #time_index="ARDATE"
                               )

es = es.entity_from_dataframe(entity_id="CUSTOMER",
                               dataframe=dfAllData_3,
                               index="CUSTOMERCODE"
                               )
   
#entities = {
 #   "esAllData3" : (alldata, "ID"),
  #  "esAllData2" : (alldata, "ID2"),
   # "esAllData" : (alldata, "ID")
 #}

#print(es.entity_dict)

def newfunc(numeric,numeric2):
    return numeric*numeric2

def absolute(column):
     return abs(column)

#Newfunc = make_trans_primitive(function=absolute,
 #                                input_types=[Numeric,Numeric],
  #                               return_type=Numeric)

#Newfunc = make_agg_primitive(function=newfunc,input_types=[Numeric,Numeric],return_type=Numeric)

#es=es.add_relationship(ft.Relationship(es["esAllData2"]["C_ID"],es["esAllData3"]["C_ID"]))

r_customer = ft.Relationship(es['CUSTOMER']['CUSTOMERCODE'],es['ARMINVOICE']['CUSTOMERCODE'])

es = es.add_relationship(r_customer)

#es=es.add_relationship(ft.Relationship(es["ARAMT_L1M"]["CUSTOMERMCODE"],es["ARMINVOICE"]["CUSTOMERMCODE"]))

r_payments = ft.Relationship(es['ARAMT_L1M']['CUSTOMERMCODE'],es['CUSTOMER']['CUSTOMERMCODE'])

es = es.add_relationship(r_payments)

print(es)

print(es["ARMINVOICE"])

#print(ft.primitives.list_primitives())

#relationships = [("esAllData2", "Index", "esAllData", "A_ID"),
#                  ("esAllData2", "C_ID", "esAllData3", "C_ID")]
#relationships = [("esAllData2", "ID2", "esAllData", "ID2"),
 #                 ("esAllData2", "ID2", "esAllData3", "ID2")]

# == The ouput of DFS is a feature matrix and the corresponding list of feature definitions. ==
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      #entities=es,
                                      #relationships=relationships,
                                      target_entity="ARAMT_L1M",
                                      #agg_primitives=["MEAN","SUM","MODE"],
                                      #agg_primitives=[Newfunc],
                                      trans_primitives=["absolute"],
                                      #trans_primitives=[],
                                      max_depth=2)
#print(feature_matrix[['MEAN(esAllData2.SUM(esAllData.Age))']].head(5))
tEnd = time.time() #計時結束
#x=[0,1,2,3,4,5,6,]
#feature_matrix.drop(feature_matrix.columns[0:1609],axis=1,inplace=True)
print(feature_matrix)
#print(feature_defs)
print(tEnd-tStart)
#print(ft.primitives.list_primitives())

#feature_matrix.to_csv('C:\\Users\\vincentkuo\\Documents\\FINAL_DATA.csv')

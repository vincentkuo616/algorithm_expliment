# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:07:45 2019

Updated on Thu Dec 19 11:57:45 2019

@author: vincentkuo
"""

import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import time 
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics   ###?算roc和auc
from sklearn.metrics import confusion_matrix
from termcolor import colored
import warnings
from datetime import datetime, timedelta
import shap
warnings.filterwarnings('ignore')
shap.initjs()

# prjRootFolder
#  ├ Code : Python Code
#  ├ Data : Dataset
#  ├ Output : Model
#  ├───@best : 本輪最佳解存在這
#  └ Report : log

save=1
save_ShapDoc=0  #Shap 值文字檔
save_ShapPlot=0 #Shap 值圖片檔
#存檔開關，0=不存，1=存
'''
A0=[9,10,11,12,13,14,15,16,25,27,28,31,33,35,38,39,45,47,52,56,58,63,67,76,79,82,84,86,87,89,91,92,93,94,95,97,98,99,100,102,103,104,105,106,108,109,111,112,115,123,130,131,142,144,147,150,153,156,157,159,160,163,164]
A1=[9,10,13,14,20,23,24,27,28,30,33,35,36,39,42,43,48,51,55,56,59,62,64,65,69,78,79,81,83,84,85,86,89,91,92,93,94,95,97,98,99,100,101,102,103,104,105,107,111,113,116,119,121,122,126,128,133,135,136,143,144,146,149,151,152,153,164]
B0=[9,10,11,13,14,16,19,20,21,25,26,29,30,32,33,36,37,40,44,49,51,53,57,60,62,63,65,66,69,72,73,75,80,83,84,85,86,87,88,90,93,94,96,97,98,99,100,102,103,104,105,107,110,111,112,116,117,118,120,122,127,129,131,132,135,136,137,139,140,141,144,146,148,152,153,154,156,158,159,162,163,164]
B1=[8,10,11,12,13,14,15,17,18,19,20,22,25,27,28,36,37,38,39,41,46,50,51,54,56,58,61,63,68,70,71,74,77,80,84,85,86,90,91,92,93,94,96,97,98,99,100,103,105,106,108,111,112,114,116,121,125,129,131,134,138,142,145,146,148,152,153,156,157,159,163]
A0=[7,8,67,78,88,102,106,107,109,112,126,135,171,213,216,217,368,375,380,385,391,403,407,428,440,475,479,494,497,499,508,512,514,517,529,531,534,536,539,551,552,553,558,559,565,566,567,569,578,579,636,637,638,639,644,645,646,647,653,654,655,660,662,663,669,671,674,676,686,704,720,721,722,727,744,745,756,757,758,759,763,764,768,777,798,803,821,828,829,861,876,883,913,914,915,927,935,672,481,574,703,572,573,533,532,560,496,495,493,706,530,934,170,169,863,9,862,166,918,218,877,917]
A0=[7,8,67,78,88,102,106,107,109,112,126,135,171,213,216,217,368,375,380,385,391,403,407,428,440,475,479,494,497,499,508,512,514,517,529,531,534,536,539,551,552,553,558,559,565,566,567,569,578,579,636,637,638,639,644,645,646,647,653,654,655,660,662,663,669,671,674,676,686,704,720,721,722,727,744,745,756,757,758,759,763,764,768,777,798,803,821,828,829,861,876,883,913,914,915,927,935,672,481,574,703,572,573,533,532,560,496,495,493,706,530,934,170,169,863,9,862,166,918,218,877,917]
A0=[7,8,67,78,88,102,106,107,109,112,126,135,171,213,216,368,375,380,385,391,403,407,428,440,475,479,494,497,499,508,517,529,531,534,536,539,551,552,553,558,559,565,566,567,569,579,636,637,638,639,644,645,646,647,653,654,655,660,662,663,669,671,674,676,686,704,720,721,722,727,744,745,756,757,758,759,763,764,768,777,798,803,821,828,829,861,876,883,913,914,915,927,935,672,481,574,703,572,573,533,532,560,496,495,493,706,530,934,170,169,863,9,862,166,918,218,877,917]
A1=[8,68,78,79,86,88,99,102,107,108,112,130,140,252,273,368,374,379,383,384,386,391,395,398,404,409,428,439,463,478,479,495,499,509,511,512,513,514,516,529,530,531,534,536,539,551,552,563,565,566,567,569,578,579,636,637,638,639,644,645,646,647,653,654,655,660,662,669,671,674,676,704,717,721,722,723,744,745,756,757,758,769,796,800,805,807,821,824,831,844,848,858,859,863,881,883,553,558,559,672,574,703,572,673,573,497,533,532,560,496,493,706,494,7,861,914,935,934,170,169,9,862,876,913,171,915,166,918,877,917,921,916,919,922,867,897,159,866,896,907,172,869,888,153]
A1=[8,68,78,79,86,88,99,102,107,108,112,130,140,252,273,368,374,379,383,384,386,391,395,404,409,428,439,463,478,479,495,499,509,511,512,513,514,529,530,531,534,536,539,551,552,563,565,566,567,569,578,579,636,637,638,639,644,645,646,647,653,654,655,660,662,669,671,674,676,704,717,721,722,723,744,745,756,757,758,769,796,800,805,807,821,824,831,844,848,858,859,863,881,883,553,558,559,672,574,703,572,673,573,497,533,532,560,496,493,706,494,7,861,935,934,169,9,862,876,915,166,918,877,921,916,922,897,159,866,896,907,869,888,153]
B0=[7,73,78,79,84,86,88,92,94,98,103,108,117,333,130,139,368,375,378,383,386,391,398,404,409,428,436,440,463,475,483,516,539,569,578,579,636,639,645,646,669,671,674,676,704,717,721,722,723,727,745,757,758,759,766,768,771,776,779,781,787,796,800,809,821,827,829,830,844,848,851,855,859,860,863,870,871,873,883,928,937]
Aset_rece06_B0=[7,73,78,79,84,86,88,92,94,98,103,108,117,333,130,139,368,375,378,383,386,391,398,404,409,428,436,440,463,475,483,516,539,569,578,579,636,639,645,646,669,671,674,676,704,717,721,722,723,727,745,757,758,759,766,768,771,776,779,781,787,796,800,809,821,827,829,830,844,848,851,855,859,860,863,870,871,873,883,928,495,498,509,512,513,514,517,529,530,536,637,638,644,647,653,654,660,663,744]
Aset_rece06_B1=[8,67,77,79,84,86,94,101,102,106,107,112,118,128,130,138,368,375,378,383,385,392,404,428,440,451,475,483,539,569,579,636,639,645,653,654,669,671,674,676,686,720,721,723,729,735,745,757,758,759,763,764,766,768,777,796,802,807,814,821,827,829,843,849,862,864,865,872,877,495,499,517,530,533,534,536,637,638,644,646,647,652,655,662,663,668,673,744,746]
B0_ADD=[8,9,67,68,135,153,159,166,169,170,171,379,380,384,385,395,407,478,479,481,493,494,496,497,499,508,511,531,532,533,534,551,552,553,558,559,560,563,565,566,567,572,573,574,655,662,672,673,686,706,720,763,769,777,803,805]
'''
B0=[8,9,67,68,73,78,79,84,86,88,92,94,98,103,108,117,130,135,139,153,159,166,169,170,171,333,368,375,378,379,380,383,384,385,386,391,395,398,404,407,409,428,436,440,463,475,478,479,481,483,493,494,495,496,497,498,499,508,509,511,512,513,514,516,517,529,530,531,532,533,534,536,539,551,552,553,558,559,560,563,565,566,567,569,572,573,574,578,579,636,637,638,639,644,645,646,647,653,654,655,660,662,663,669,671,672,673,674,676,704,717,721,722,723,727,744,745,757,758,759,766,768,771,776,779,781,787,796,800,809,821,827,829,830,844,848,851,855,859,860,863,870,871,873,883,928]
Aset_rece06_B1=[8,67,77,79,84,86,94,101,102,106,107,112,118,128,130,138,368,375,378,383,385,392,404,428,440,451,475,483,539,569,579,636,639,645,653,654,669,671,674,676,686,720,721,723,729,735,745,757,758,759,763,764,766,768,777,796,802,807,814,821,827,829,843,849,862,864,865,872,877,495,499,517,530,533,534,536,637,638,644,646,647,652,655,662,663,668,673,744,746]
B1_ADD=[7,9,78,88,99,108,109,153,159,166,169,170,171,407,409,463,478,479,481,493,494,496,497,508,509,511,512,513,514,529,531,532,551,552,553,558,559,560,563,565,566,567,572,573,574,578,703,704,706,717]
B1=[7,8,9,67,77,78,79,84,86,88,94,99,101,102,106,107,108,109,112,118,128,130,138,153,159,166,169,170,171,368,375,378,383,385,392,404,407,409,440,451,463,475,478,479,481,483,493,494,495,496,497,499,508,509,511,512,513,514,517,529,530,531,532,533,534,536,539,551,552,553,558,559,560,563,565,566,567,569,572,573,574,578,579,636,637,638,639,644,645,646,647,652,653,654,655,662,663,668,669,671,673,674,676,686,703,704,706,717,720,721,723,729,735,744,745,746,757,758,759,763,764,766,768,777,796,802,807,814,821,827,829,843,849,862,864,865,872,877]
A0 = [7,8,67,78,88,102,106,107,109,112,126,135,171,213,216,368,375,380,385,391,403,407,428,440,475,479,494,497,499,508,517,529,531,534,536,539,551,552,553,558,559,565,566,567,569,579,636,637,638,639,644,645,646,647,653,654,655,660,662,663,669,671,674,676,686,704,720,721,722,727,744,745,756,757,758,759,763,764,768,777,798,803,821,828,829,861,876,883,913,914,915,927,935,672,481,574,703,572,573,533,532,560,496,495,493,706,530,934,170,169,863,9,862,166,918,218,877,917]
A1 = [8,78,79,86,88,99,102,107,108,112,130,140,252,273,368,374,379,383,384,386,391,395,404,409,428,439,463,478,479,495,499,509,511,512,513,514,529,530,531,534,536,539,551,552,563,565,566,567,569,578,579,636,637,638,639,644,645,646,647,653,654,655,660,662,669,671,674,676,704,717,721,722,723,744,745,756,757,758,769,796,800,805,807,821,824,831,844,848,858,859,863,881,883,553,558,559,672,574,703,572,673,573,497,533,532,560,496,493,706,494,7,861,935,934,169,9,862,876,915,166,918,877,921,916,922,897,159,866,896,907,869,888,153]
#Aset_rece06_B1.extend(B1_ADD)
A0 = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,834,835,836,837]
'''
A1.remove(917)
A1.remove(170)
A1.remove(171)
A1.remove(172)
A1.remove(398)
A1.remove(516)
A1.remove(867)
A1.remove(913)
A1.remove(914)
A1.remove(919)
###
A1.remove(915)
A1.remove(921)
'''
A1=[13,19,42,44,47,107,119,138,143,148,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218]
A1_2=[13,19,42,44,47,107,119,143,154,160,161,163,171,172,174,175,176,177,178,179,180,182,183,185,186,188,189,190,191,192,194,195,196,197,199,201,206,214,215,216,217,218]
A1_2_DEL=[165,166,167,168,169,170,173,181,184,187,193,198,200,202,203,204,205,208,209,210,211,212,213,215]
#A1_2_DEL=[119,153,154,155,156,157,158,159,162,164,165,166,167,168,169,170,173,181,184,187,193,198,200,202,203,204,205,208,209,210,211,212,213,215]
A1_20=[15,19,53,213,259,292,300,302,303,304,305,307,308,309,310,311,312,313,314,316]
A1_49=[15,19,53,112,152,213,233,238,253,259,265,268,273,275,277,278,280,281,284,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316]
A0_38F = [15,19,53,213,259,292,300,302,303,304,305,307,308,309,310,311,312,313,314,316,11,12,13,14,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
A1_69F = [11,13,16,18,19,20,21,29,36,44,45,50,52,54,62,70,76,84,85,86,96,104,107,109,111,112,114,126,133,137,146,152,164,178,190,197,199,209,215,225,228,229,234,235,236,243,246,251,263,264,265,267,275,296,299,302,311,320,331,335,341,348,362,363,366,367,373,375,376]
fname=['DIFF_L1M_MED_COM','PAYMENTTYPE_MODE_CASH','OCBetrayer_AR_CNT_ALL_RATIO','CREDITTOTAL_L6M_MAX','RECEDELAY14_OLHYTOALL_RATIO','Pattern_6M_ARCNT.1_cont','RECEDELAY7_OL4Q_CNT','Pattern_6Q_ARAMT.perQ_cont','ARAMT_ALL_AVG','ARAMT_L6M_STD','ARAMTRANK_OL3M_MED_A0','OCCreditUsed_LL6M_AVG','OCCreditUsed_LL3M_AVG_adjust','SRAMT_L1Y_SUM','RECEDELAY14_OLHY_YN','RECEDELAY3_OL2Q_CNT','RECEDELAY7_OL3Q_CNT','RECEDELAY7_OLHYTOALL_CNT','DIFF_L1M_STD_COM','Pattern_6Q_ARCNT.3_cont','DIFF_L1W_MAX_COM','DIFF_L1M_AVG_COM','DIFF_L1W_AVG_COM','DIFF_L1W_MED_COM','DIFF_L1M_MAX_COM','DIFF_L1W_MIN_COM','DIFF_L1M_MIN_COM','ARAMT_OL2Q_AVG','ARAMT_OL3Q_AVG','ARAMT_OL4Q_AVG','ARAMTRANK_OL2Q_AVG','ARAMTRANK_OL3Q_AVG','ARAMTRANK_OL4Q_AVG','OCCOAMT_LL1M_AVG','OCCOAMT_LL3M_AVG','Pattern_6M_ARAMT.4.7w_cont','Pattern_6Q_ARAMT.17w_cont','ARAMT_Cred_L1M_Ratio']
#fname=['DIFF_L1M_MAX_COM','DIFF_L1W_MAX_COM','PAYMENTTYPE_MODE_MONTH','DIFF_L1M_AVG_COM','DIFF_L1M_MIN_COM','DIFF_OL2M_MIN_COM','DIFF_OL3W_MAX_COM','DIFF_OL2W_AVG_COM','PAYMENTTYPE_MODE_CASH','OC_Pass_L6M_RATIO','OCCred_AR_CNT_L6M_Ratio','OCCred_AR_CNT_L1Y_Ratio','ARAMTRANK_OL2M_MED','OCCreditUsed_LL6M_AVG_v2','OCCREDIT_L6M_CNT_RESULTY','ARAMT_Cred_L1M_Ratio','ARAMTRANK_OL3M_SUM_A1','ARAMT_L3M_SK','ocAcceptedLock_L3M_YN','NOTEDIFF61_L6M_MED','OCCO_ARAMT_AVG_LL6M_RATIO','ocReapeatedRatio_L1Y','OCCredit_AR_CNT_L1M_RATIO','GGCNT_L3MxL6M_RATIO','NOTEDIFF61_L3M_AVG','NOTEDIFF61_L1Y_AVG','BUYBOMBAVG9_RATIO','BUYBOMBCREDNORMA5_L3M_RATIO','ISBADNOTEHAPPEN_L1M','ARAMT_L3M_L1Y_STD','ARAMTRANK_OL2M_MAX','ARAMTRANK_L1Y_STD','ARAMTRANK_L6M_AVG_A1','ARAMTRANK_OL4Q_MAX','ARAMT_L6M_MAX','BUYBOMBCREDNORMA5_L3Y_RATIO','BUYBOMBAVG6_RATIO','ARAMTperGG_genCustSizeG_L1Y','ARAMTRANK_OL3M_AVG','ARAMTRANK_L3M_AVG_A1','ARAMT_L1YxL3Y_AVG_RATIO','OCARRATIO_ALL_CNT','ARAMTRANK_OL4Q_AVG_A1','ARAMTRANK_OL2M_MED_A1','ARAMTRANK_L3Y_STD_A1','ARAMTRANK_L6M_STD_A1','RECEMODE_OL234QxOL2Y_RATD','ARAMTRANKgen_L1MxL1Y_AVGD','ARAMTRANK_OL4Q_SUM','BUYBOMBCREDIT7_L3M_RATIO','BUYBOMBAVG7_RATIO','OCAMT_LL3M_AVG','ARAMT_L6M_SK','ARAMT_L3MxL3Y_AVG_RATIO','RECETYPEMODE_OL2Y_RATIO','CREDITTOTAL_L1Y_MAX','RECETYPEMODE_OLHY_RATIO','RECEDELAY1_OLHYTOALL_CNT','ARAMTRANKgen_L1MxL1Y_AVGDA1','RECEDELAY7_OLHY_CNT','ARAMT_perM','ARAMTRANK_L1Y_MED','BUYBOMBCREDIT5_L1Y_RATIO','ARAMT_L3M_CV','ARCNT_genFY_L1YxALL_RATIO','OCCOAMT_LALL_AVG','BUYBOMBCREDNORMA9_L3M_RATIO','BUYBOMBCREDIT9_L3M_RATIO','OCAMT_LL3Y_AVG']

#A0_38F = [15,19,53,213,259,292,300,302,303,304,305,307,308,309,310,311,313,314,316,20,23,26,27,28,29,30]
#fname=['DIFF_L1M_MED_COM','PAYMENTTYPE_MODE_CASH','OCBetrayer_AR_CNT_ALL_RATIO','CREDITTOTAL_L6M_MAX','RECEDELAY14_OLHYTOALL_RATIO','Pattern_6M_ARCNT.1_cont','RECEDELAY7_OL4Q_CNT','Pattern_6Q_ARAMT.perQ_cont','ARAMT_ALL_AVG','ARAMT_L6M_STD','ARAMTRANK_OL3M_MED_A0','OCCreditUsed_LL6M_AVG','OCCreditUsed_LL3M_AVG_adjust','SRAMT_L1Y_SUM','RECEDELAY14_OLHY_YN','RECEDELAY3_OL2Q_CNT','RECEDELAY7_OLHYTOALL_CNT','DIFF_L1M_STD_COM','Pattern_6Q_ARCNT.3_cont','ARAMT_OL2Q_AVG','ARAMTRANK_OL2Q_AVG','OCCOAMT_LL1M_AVG','OCCOAMT_LL3M_AVG','Pattern_6M_ARAMT.4.7w_cont','Pattern_6Q_ARAMT.17w_cont','ARAMT_Cred_L1M_Ratio']
#A1_69F = [11,13,16,18,19,20,21,29,36,44,45,52,54,62,70,76,84,85,86,96,104,109,114,126,133,137,146,152,164,178,190,197,209,215,228,229,234,235,236,243,246,251,263,264,267,275,296,299,302,311,320,331,335,348,362,363,366,367]
#fname=['DIFF_L1M_MAX_COM','DIFF_L1W_MAX_COM','PAYMENTTYPE_MODE_MONTH','DIFF_L1M_AVG_COM','DIFF_L1M_MIN_COM','DIFF_OL2M_MIN_COM','DIFF_OL3W_MAX_COM','DIFF_OL2W_AVG_COM','PAYMENTTYPE_MODE_CASH','OC_Pass_L6M_RATIO','OCCred_AR_CNT_L6M_Ratio','ARAMTRANK_OL2M_MED','OCCreditUsed_LL6M_AVG_v2','OCCREDIT_L6M_CNT_RESULTY','ARAMT_Cred_L1M_Ratio','ARAMTRANK_OL3M_SUM_A1','ARAMT_L3M_SK','ocAcceptedLock_L3M_YN','NOTEDIFF61_L6M_MED','OCCO_ARAMT_AVG_LL6M_RATIO','ocReapeatedRatio_L1Y','GGCNT_L3MxL6M_RATIO','BUYBOMBAVG9_RATIO','BUYBOMBCREDNORMA5_L3M_RATIO','ISBADNOTEHAPPEN_L1M','ARAMT_L3M_L1Y_STD','ARAMTRANK_OL2M_MAX','ARAMTRANK_L1Y_STD','ARAMTRANK_L6M_AVG_A1','ARAMTRANK_OL4Q_MAX','ARAMT_L6M_MAX','BUYBOMBCREDNORMA5_L3Y_RATIO','ARAMTperGG_genCustSizeG_L1Y','ARAMTRANK_OL3M_AVG','ARAMT_L1YxL3Y_AVG_RATIO','OCARRATIO_ALL_CNT','ARAMTRANK_OL4Q_AVG_A1','ARAMTRANK_OL2M_MED_A1','ARAMTRANK_L3Y_STD_A1','ARAMTRANK_L6M_STD_A1','RECEMODE_OL234QxOL2Y_RATD','ARAMTRANKgen_L1MxL1Y_AVGD','ARAMTRANK_OL4Q_SUM','BUYBOMBCREDIT7_L3M_RATIO','OCAMT_LL3M_AVG','ARAMT_L6M_SK','ARAMT_L3MxL3Y_AVG_RATIO','RECETYPEMODE_OL2Y_RATIO','CREDITTOTAL_L1Y_MAX','RECETYPEMODE_OLHY_RATIO','RECEDELAY1_OLHYTOALL_CNT','ARAMTRANKgen_L1MxL1Y_AVGDA1','RECEDELAY7_OLHY_CNT','ARAMTRANK_L1Y_MED','BUYBOMBCREDIT5_L1Y_RATIO','ARAMT_L3M_CV','ARCNT_genFY_L1YxALL_RATIO','OCCOAMT_LALL_AVG']
A1_Aset=[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186]

for i in A1_2_DEL: A1.remove(i)
sel_f = A0_38F # 挑選之特徵集
feature_Y=2 #Y是第幾個欄位

def main():
    txt_name="VI_Y200310_A1" #僅供LOG檔名使用    
    
    #資料結構請參考上面
    prjRootFolder="C://Users//vincentkuo//Documents//vincent_TW//"
    #TrainSet = pd.read_csv(prjRootFolder+"Train_Data_rename.csv",encoding='utf-8')
    #TestSet = pd.read_csv(prjRootFolder+"Test_Data_rename.csv",encoding='utf-8')
    reportPath=prjRootFolder+"Report//XGBoost_"+txt_name+"_log"+getDatetimeStr()+".txt"
    
    allDataSet = pd.read_csv(prjRootFolder+"A0_IV_Y200117.csv",encoding='utf-8')
    #allDataSet = pd.read_csv(prjRootFolder+"A1_online_use_test.csv",encoding='utf-8')
    TrainSet = allDataSet.groupby("SAMPLE").get_group("TRAIN")
    TestSet = allDataSet.groupby("SAMPLE").get_group("TEST")
    #allDataSet = TrainSet.append(TestSet)
    
    #將資料分組
    TrainSet_A0 = pd.read_csv(prjRootFolder+"A0_IV_Y200117_8%.csv",encoding='utf-8')
    TrainSet_A1 = TrainSet.groupby("GROUPA").get_group("A1")
    #TrainSet_B0 = pd.read_csv(prjRootFolder+"B0_Train_Data_Resampling_8%.csv",encoding='utf-8')
    #TrainSet_B1 = pd.read_csv(prjRootFolder+"B1_Train_Data_Resampling_8%.csv",encoding='utf-8')
    TestSet_A0 = allDataSet.groupby("SAMPLE").get_group("TEST").groupby("GROUPA").get_group("A0")
    TestSet_A1 = allDataSet.groupby("SAMPLE").get_group("TEST").groupby("GROUPA").get_group("A1")
    #TestSet_B0 = TestSet.groupby("GROUPB").get_group("B0")
    #TestSet_B1 = TestSet.groupby("GROUPB").get_group("B1")
    
    TrainData = TrainSet_A0 # 要跑的訓練集
    TestData  = TestSet_A0  # 要跑的測試集

    #存檔開關
    if save == 1:
        titleStr="trainSetName,index,costTime,featureSet,max_depth,gamma,subsample,scale_pos_weight,eta,min_child_weight,estimators,TN,FP,FN,TP,Precision,Recall,Specificity,Sensitivity,Accuracy,F1-Score,fileName,TimeStamp"
        with open(reportPath, "a") as myfile:
            myfile.write(titleStr+"\n")
        
    '''
    https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
    '''
    index=0
    
    #loop setting
    #========================================================
    #最大深度 建議: [3-10]
    depthMin=3
    depthMax=12
    depthStep=20
    #分裂後損失函數大於此閾值才會長子結點，增加:避免overfitting 建議: [0-0.2]
    gammaMin=0
    gammaMax=0.3
    gammaStep=1
    #對於每棵樹隨機採樣的比例，降低:避免overfitting；過低:underfitting 建議: [0.5-0.9]
    subsampleMin=0.8
    subsampleMax=1.05
    subsampleStep=1
    #colsample_bytree 控制每顆樹隨機採樣的列數的佔比 建議: [0.5-0.9]
    cbMin=0.7
    cbMax=1.05
    cbStep=3
    #learning rate
    etaMin=0.3
    etaMax=0.51
    etaStep=1
    #min_child_weight 決定最小葉子節點樣本權重和，加權和低於這個值時，就不再分裂產生新的葉子節點 建議: [1]
    mcwMin=0
    mcwMax=2
    mcwStep=10
    #boost迭代次數
    itersMin=1000
    itersMax=1400
    itersStep=1000
    #========================================================

    
    #A0_LEN=TrainSet_A0.shape[1]-11
    #loopStart
    
    logRoof()
    print("   [ test Loop ]") 
    logFloor()
    
    bM=bestModel("", "", 0, 0, {}, {}, xgb.core.Booster(), {}, 0)

    #for featureSet in range (0,4):
     #   copyf=sel_f.copy()
      #  del copyf[-1-featureSet]
    for featureSet in range (0,1):
        endat=sel_f
        if featureSet==0 :
            feature_Desc="xgboost_A0_TW_COM5_Local_20"
            trainSetName="xgboost_A0_TW_COM5_Local_20" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,feature_Y].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
        else :
            endat = A1_49
            feature_Desc="xgboost_A0_TW_COM5_Local_49"
            trainSetName="xgboost_A0_TW_COM5_Local_49" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,feature_Y].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
        '''
        elif featureSet==2 :
            TrainData = pd.read_csv(prjRootFolder+"B0_Train_Data_Resampling_AS_13%.csv",encoding='utf-8')
            feature_Desc="xgboost_B0_Local_AS"
            trainSetName="xgboost_B0_Local_AS" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,feature_Y].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
        
        else :
            endat=21
            feature_Desc="All+OneHot"
            trainSetName="xgboost_test_3" #文件的分類
            X_train = TrainData.iloc[:,endat].values
            y_train = TrainData.iloc[:,feature_Y].values
            y_train = transformYNto10(y_train)  #檢查 Y 值
            X_test = TestData.iloc[:,endat].values
            y_test = TestData.iloc[:,feature_Y].values
            y_test = transformYNto10(y_test)    #檢查 Y 值
        '''
        #fname=TestData.iloc[0:0,endat].columns.values
        
        for p_depth in np.arange (depthMin, depthMax, depthStep): #max_depth
            for p_gamma in np.arange(gammaMin, gammaMax, gammaStep): #gamma
                for p_ss in np.arange(subsampleMin, subsampleMax, subsampleStep):
                    for p_cb in np.arange(cbMin, cbMax, cbStep):
                        for p_eta in np.arange (etaMin, etaMax, etaStep): 
                            for p_mcw in np.arange (mcwMin, mcwMax, mcwStep):
                                for iters in np.arange (itersMin, itersMax, itersStep):
                                    index += 1
                                    print("[{}] {} : {} x {}, {}".format(index,feature_Desc,X_train.shape[0],X_train.shape[1],getTimeNow()))
                                    print("    >> max_depth:{}, gamma:{}, ss:{}, cb:{}, eta:{}, mcw:{}, estimators:{}".format(p_depth,p_gamma,p_ss,p_cb,p_eta,p_mcw,iters))
                                    params = {
                                        'booster': 'gbtree',
                                        'objective': 'binary:logistic',
                                        'gamma': p_gamma,
                                        'max_depth': p_depth,
                                        'lambda': 2,
                                        'subsample': p_ss,
                                        'colsample_bytree': p_cb, 
                                        'min_child_weight': p_mcw, 
                                        'silent': 1,
                                        'eta': p_eta, 
                                        'nthread': -1,
                                        'eval_metric' : 'error',
                                        'scale_pos_weight': 1, #如果出現嚴重的不平衡，則應使用大於0的值，因為它有助於加快收斂速度。 負樣本個數/正樣本個數 
                                        #上面這條有的人說可以調，有的人說調了沒用，所以請自行參考斟酌
                                    }
                                    plst = params.items()
                                    dtrain = xgb.DMatrix(X_train, y_train, feature_names=fname)
                                    #dtrain = xgb.DMatrix(X_train, y_train)
                                    timestp = getDatetimeStr()
                                    filePath = prjRootFolder+"Output//"+timestp+".pkl"
                                    lstime = time.time()
                                    
                                    #TRAIN
                                    model = xgBoost_train(plst, dtrain, iters, filePath,X_test,fname)
                                    #TEST
                                    mS = xgBoost_testForLoop(X_test,y_test,filePath,fname)
                                    
                                    letime=time.time()
                                    costTimeTrain=round(letime-lstime,2)
                                    print("  >> cost: "+str(costTimeTrain)+"s")
                                    resultStr="{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(trainSetName,index,costTimeTrain,featureSet,p_depth,p_gamma,p_ss,p_cb,p_eta,p_mcw,iters,mS.toString(),timestp,getTimeNow())
                                    if save == 1:
                                        with open(reportPath, "a") as myfile:
                                            myfile.write(resultStr+"\n")
                                        
                                    if mS.f1s>bM.f1s:
                                        bM=bestModel(filePath, timestp, index, mS.f1s, X_test, y_test, model, fname, endat)
                                    
    #playsound(prjRootFolder+'//Code//CNoc.mp3')
    logRoof()
    if bM.index!=0:
        print(">> Loop End @ "+str(index))
        print(">> Best Sel @ {}".format(bM.index))
        xgBoost_testFortheBest(bM.X_test, bM.y_test, bM.filePath, bM.fname, prjRootFolder)
        #getFinalResultForThisRound(prjRootFolder, bM, allDataSet)
    else:
        print("No Best Sel @")
    #fs=pd.Series(bM.model.get_fscore()).sort_values(ascending=False)
    #print(fs)
    logFloor()

def xgBoost_train(plst, dtrain, num_rounds,filePath,X_test,fname):#train_data,nu,kernel,gamma
    model = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(X_test, feature_names=fname)
    ans = model.predict(dtest)
    print(ans)
    with open(filePath, 'wb') as model_file:
        joblib.dump(model, model_file)
    return model

def xgBoost_testForLoop(X_test,y_test,filePath,fname):
    with open(filePath, 'rb') as f:
        model = joblib.load(f)
        # pred the test data
        dtest = xgb.DMatrix(X_test, feature_names=fname)
        #dtest = xgb.DMatrix(X_test)
        ans = model.predict(dtest)
        print(ans)
        cm = confusion_matrix(y_test, (ans>0.5))
        mS = getCMresults(cm)
        print(colored('  The Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> Recall = ', 'blue'), colored(toPercentage(mS.recall), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))
    return mS

def xgBoost_testFortheBest(X_test,y_test,filePath,fname,prjRootFolder):
    with open(filePath, 'rb') as f:
        model = joblib.load(f)
        # pred the test data
        dtest = xgb.DMatrix(X_test, feature_names=fname)
        #dtest = xgb.DMatrix(X_test)
        ans = model.predict(dtest)
        cm = confusion_matrix(y_test, (ans>0.5))
        mS=getCMresults(cm)
        print(colored('  The Best Confusion Matrix is: ', 'red'),'\n', cm)
        print(colored('  >> ACC = ', 'blue'), colored(toPercentage(mS.accuracy), 'blue'))
        print(colored('  >> Recall = ', 'blue'), colored(toPercentage(mS.recall), 'blue'))
        print(colored('  >> F1-score = ', 'blue'), colored(toPercentage(mS.f1s), 'blue'))
        plot_importance(model, max_num_features=20)
        if save_ShapPlot == 1:
            plt.savefig(prjRootFolder+"@best//XGB_ImportancePlot.png",bbox_inches='tight')
        plt.show()
        
    return mS
    

def getFinalResultForThisRound(prjRootFolder, bM, allDataSet):
    outputFolder=prjRootFolder+"@best//"
    X_importance=bM.X_test
    X_all = allDataSet.iloc[:,bM.endat].values
    pk_all = allDataSet.iloc[:,0:8].values
    dtest = xgb.DMatrix(X_all, feature_names=bM.fname)
    ans = bM.model.predict(dtest)
    test = np.append(pk_all,np.array([ans]).transpose(),axis=1)
    resultColumnsName = np.append(allDataSet.columns.values[0:8],'Probability')
    result = pd.DataFrame(test,columns=resultColumnsName)
    #https://liuyanguu.github.io/post/2019/07/18/visualization-of-shap-for-xgboost/
    #https://shap.readthedocs.io/en/latest/
    #http://sofasofa.io/tutorials/shap_xgboost/
    #https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html
    #https://medium.com/ai-academy-taiwan/explain-your-machine-learning-model-by-shap-part-1-228fb2a57119
    explainer = shap.TreeExplainer(bM.model)
    shap_values = explainer.shap_values(X_importance)
    
    #嫌圖太大改這邊
    if save_ShapPlot == 1:
        shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=len(bM.endat), show=False)
        plt.savefig(outputFolder+bM.fileName+"_shap_summary.png",bbox_inches='tight')
        #這邊存檔會有圖形疊加的問題
        #shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, plot_type='bar', max_display=len(bM.endat), show=False)
        #plt.savefig(outputFolder+bM.fileName+"_shap_summary_bar.png",bbox_inches='tight')
        graph = xgb.to_graphviz(bM.model, num_trees=1, **{'size': str(10)})
        graph.render(directory=outputFolder,filename=str(bM.fileName)+'_xgb.dot')
        #bM.model.save_model('{}_xgb.model'.format(bM.fileName))
    else:
        shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, max_display=len(bM.endat), show=True)
        shap.summary_plot(shap_values, X_importance, feature_names=bM.fname, plot_type='bar', max_display=len(bM.endat), show=True)
  
    fs=pd.Series(bM.model.get_fscore()).sort_values(ascending=False)


    if save_ShapDoc == 1:
       svpd = pd.DataFrame(shap_values)
       result.to_csv("{}//{}_Result.csv".format(outputFolder,bM.fileName))
       svpd.to_csv("{}//{}_ResultSHAP.csv".format(outputFolder,bM.fileName))
       fs.to_csv("{}//{}_FeatureImportance.txt".format(outputFolder,bM.fileName))
    

def esttimeplz(loopCount,sec):
    est=int(loopCount)*sec   #estimate time
    d = datetime.now()+ timedelta(seconds = est)
    return '{0:%Y/%m/%d %H:%M:%S}'.format(d)

def getDatetimeStr():
    d = datetime.now()
    str='{:%Y%m%d%H%M%S}'.format(d)
    return str
 

def getTimeNow():
    return '{0:%Y/%m/%d %H:%M:%S}'.format(datetime.now())

def getCMresults(cm):
    C_matrix = cm
    tn=C_matrix[0, 0]
    fp=C_matrix[0, 1]
    fn=C_matrix[1, 0]
    tp=C_matrix[1, 1]
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision= tp / (tp + fp)
    accuracy=(tp + tn) / (tp + fn + fp + tn)
    recall= tp / (tp+fn)
    f1s=round(2*tp/(2*tp+fn+fp),6)

    return mScore(tn, fp, fn, tp, precision, recall, specificity,sensitivity, accuracy, f1s)
    
def toPercentage(floatNum):
    return "{percent:.4%}".format(percent=floatNum)

def logS(title):
    print("\n ========================================== ")
    print("   [ "+title+" Start ] "+getTimeNow())
    print("\n")    
    global Stime
    Stime=time.time()
    #print(" ===================================== ")

def logE(title):
    Etime = time.time()
    global Stime
    #print("\n ===================================== ")
    print("\n")
    print("   Cost: "+"{}".format(round(Etime-Stime,2))+"s" )
    print("   [ "+title+" End ] "+getTimeNow())
    print(" ========================================== ")

def log(title):
    print("\n ========================================== ")
    print("   [ "+title+" ] "+getTimeNow())
    print(" ========================================== ")
    
def logRoof():
    print("\n ========================================== ")

def logFloor():
    print("   "+getTimeNow())
    print(" ========================================== ")

def transformYNto10(yn):
    try:
        yn[yn=='Y']=1
        yn[yn=='N']=0
        yn = yn.astype(np.int64)
        return yn
    except:
        print("Something Wrong at transformYNto10")
    

class bestModel:
    def __init__(self, filePath, fileName, index, f1s, Xt, yt, model, fname, endat):
        self.filePath=filePath
        self.index=index
        self.f1s=f1s
        self.X_test=Xt
        self.y_test=yt
        self.fileName=fileName
        self.model=model
        self.fname=fname
        self.endat=endat

class mScore:
    def __init__(self, tn, fp, fn, tp, precision, recall, specificity,sensitivity, accuracy, f1s):
        self.tn=tn
        self.fp=fp
        self.fn=fn
        self.tp=tp
        self.precision=precision
        self.recall=recall
        self.specificity=specificity
        self.sensitivity=sensitivity
        self.accuracy=accuracy
        self.f1s=f1s        
    def toString(self):
        return "{},{},{},{},{},{},{},{},{},{}".format(self.tn, self.fp, self.fn, self.tp, self.precision, self.recall, self.specificity,self.sensitivity, self.accuracy, self.f1s)
    
if __name__ == '__main__':
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    main()
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")    
    
    


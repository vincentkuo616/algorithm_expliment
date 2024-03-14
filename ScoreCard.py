# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:51:42 2020

@author: vincentkuo
"""

"""
參考
https://zhuanlan.zhihu.com/p/59585403
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#导入数据 
a=open('//twpc19020/c$/Users/syn006/Desktop/HungYu/HW/creditCard/cs-test.csv')
b=open('//twpc19020/c$/Users/syn006/Desktop/HungYu/HW/creditCard/cs-training.csv')
train=pd.read_csv(b)
test=pd.read_csv(a)
#检查数据各字段信息
train.info()
train1=train.copy()

#将英文列名转为中文列名
states={'SeriousDlqin2yrs':'好坏客户',
        'RevolvingUtilizationOfUnsecuredLines':'可用额度比值',
        'age':'年龄',
        'NumberOfTime30-59DaysPastDueNotWorse':'逾期30-59天笔数',
        'DebtRatio':'负债率',
        'MonthlyIncome':'月收入',
        'NumberOfOpenCreditLinesAndLoans':'信贷数量',
        'NumberOfTimes90DaysLate':'逾期90天笔数',
        'NumberRealEstateLoansOrLines':'固定资产贷款量',
        'NumberOfTime60-89DaysPastDueNotWorse':'逾期60-89天笔数',
        'NumberOfDependents':'家属数量'}

#使用rename函数列的重列名
train.rename(columns=states,inplace=True)
print(train.head())

#观察原始数据集中标签的分布情况
print(train['好坏客户'].value_counts())

print('月收入缺失比:{:.2%}'.format(train['月收入'].isnull().sum()/train.shape[0]))
print('家属数量缺失比:{:.2%}'.format(train['家属数量'].isnull().sum()/train.shape[0]))

#对月收入采用平均数填充
train['月收入']=train['月收入'].fillna( train['月收入'].mean())
train=train.dropna()
train.info()
"""outlier"""
# 建立画板和画纸
fig=plt.figure(figsize=(20,15))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)
ax1.boxplot([train['可用额度比值'],train['负债率']])
ax2.boxplot(train['年龄'])
ax3.boxplot([train['逾期30-59天笔数'],train['逾期60-89天笔数'],train['逾期90天笔数']])
ax4.boxplot([train['信贷数量'],train['固定资产贷款量']])
#设置坐标轴格式
ax1.set_xticklabels(['可用额度比值','负债率'], fontsize=20)
ax2.set_xticklabels(['年龄'],fontsize=20)
ax3.set_xticklabels(['逾期30-59天笔数','逾期60-89天笔数','逾期90天笔数'], fontsize=20)
ax4.set_xticklabels(['信贷数量','固定资产贷款量'], fontsize=20)
plt.show()

# 过滤异常值
train=train[train['年龄']>18]
train=train[train['年龄']<100]
train=train[train['逾期30-59天笔数']<80]
train=train[train['逾期60-89天笔数']<80]
train=train[train['逾期90天笔数']<80]
train=train[train['固定资产贷款量']<50]
train.shape
"""探索性分析"""
#将年龄分组
age_cut=pd.cut(train['年龄'],5)
age_cut_group=train['好坏客户'].groupby(age_cut).count()
#求分组下的坏客户数
#1代表怀坏客户，sum()对于数据中等于1的数据求和
age_cut_group1=train['好坏客户'].groupby(age_cut).sum()
#连接两个表
cardDf1=pd.merge(pd.DataFrame(age_cut_group),pd.DataFrame(age_cut_group1),left_index=True,right_index=True)
cardDfdict={'好坏客户_x':'总客户数','好坏客户_y':'坏客户数'}
cardDf1.rename(columns=cardDfdict,inplace=True)
print(cardDf1)
#增加一列好客户数
cardDf1.insert(2,'好客户数',cardDf1['总客户数']-cardDf1['坏客户数'])
#增加一列坏客户占比
cardDf1.insert(2,'坏客户占比',cardDf1['坏客户数']/cardDf1['总客户数'])
cardDf_100=cardDf1['坏客户占比']*100
ax1=cardDf1[["好客户数","坏客户数"]].plot.bar(figsize=(10,5))
ax1.set_xticklabels(cardDf1.index,rotation=0)
ax1.set_ylabel('客户数')
ax1.set_title('年龄与好坏客户数分布图')
ax1=cardDf_100.plot(linewidth=2,marker='o',secondary_y=True,color='r',linestyle='dashed')
plt.xlim([-0.5,4.5])
plt.show()
"""相關係數"""
#查看各字段与坏客户的相关系数
import seaborn as sns
corr=train.corr()
corr['好坏客户'].sort_values(ascending=False)
#数据相关系数的一半
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig=plt.figure()
#建立画纸
ax1=fig.add_subplot(1,1,1)
fig.set_size_inches(15,15)
#使用heatmap胜利图
sns.heatmap(corr, mask=mask, ax=ax1,square=False,annot=True,linewidths=.5)
plt.xticks(fontsize=10,color='black',rotation=45)
plt.yticks(fontsize=10,color='black',rotation=45)
plt.show()

"""WOE"""
bins1=4
cut1=pd.qcut(train['可用额度比值'],bins1,labels=False)
bins2=8
cut2=pd.qcut(train["年龄"],bins2,labels=False)
bins3=[-1,0,1,3,5,13]
cut3=pd.cut(train["逾期30-59天笔数"],bins3,labels=False)
bins4=3
cut4=pd.qcut(train["负债率"],bins4,labels=False)
bins5=4
cut5=pd.qcut(train["月收入"],bins5,labels=False)
bins6=4
cut6=pd.qcut(train["信贷数量"],bins6,labels=False)
bins7=[-1, 1, 3,5, 20]
cut7=pd.cut(train["逾期90天笔数"],bins7,labels=False)
bins8=[-1, 0,1,2, 3, 33]
cut8=pd.cut(train["固定资产贷款量"],bins8,labels=False)
bins9=[-1, 0, 1, 3, 12]
cut9=pd.cut(train["逾期60-89天笔数"],bins9,labels=False)
bins10=[-1, 0, 1, 2, 3, 5, 21]
cut10=pd.cut(train["家属数量"],bins10,labels=False)
#print(cut7)
#print('==================================')

rate=train["好坏客户"].sum()/(train["好坏客户"].count()-train["好坏客户"].sum())
def get_woe_data(cut):
    grouped=train["好坏客户"].groupby(cut,as_index = True).value_counts()
    woe=np.log(grouped.unstack().iloc[:,1]/grouped.unstack().iloc[:,0]/rate)
    return woe
cut1_woe=get_woe_data(cut1)
cut2_woe=get_woe_data(cut2)
cut3_woe=get_woe_data(cut3)
cut4_woe=get_woe_data(cut4)
cut5_woe=get_woe_data(cut5)
cut6_woe=get_woe_data(cut6)
cut7_woe=get_woe_data(cut7)
cut8_woe=get_woe_data(cut8)
cut9_woe=get_woe_data(cut9)
cut10_woe=get_woe_data(cut10)

fig,axes=plt.subplots(5,2)
fig.set_size_inches(10,25)
ax1=cut1_woe.plot(linewidth=2,marker='o',title = 'WOE值随可用额度比值的变化关系',ax=axes[0,0])
ax2=cut2_woe.plot(linewidth=2,marker='o',title = 'WOE值随年龄的变化关系',ax=axes[0,1])
ax3=cut3_woe.plot(linewidth=2,marker='o',title = 'WOE值与逾期30-50天的变化关系',ax=axes[1,0])
ax4=cut4_woe.plot(linewidth=2,marker='o',title = 'WOE值与负债率率变化关系',ax=axes[1,1])
ax5=cut5_woe.plot(linewidth=2,marker='o',title = 'WOE值与月收入关系',ax=axes[2,0])
ax6=cut6_woe.plot(linewidth=2,marker='o',title='WOE值与信贷数量关系',ax=axes[2,1])
ax7=cut7_woe.plot(linewidth=2,marker='o',title='WOE值与逾期90天的关系',ax=axes[3,0])
ax8=cut8_woe.plot(linewidth=2,marker='o',title='WOE值与固定资产的关系',ax=axes[3,1])
ax9=cut9_woe.plot(linewidth=2,marker='o',title='WOE值与逾期60-89天的关系',ax=axes[4,0])
ax10=cut10_woe.plot(linewidth=2,marker='o',title='家属数量',ax=axes[4,1])
plt.show()
"""IV"""
def get_IV_data(cut,cut_woe):
    grouped=train['好坏客户'].groupby(cut,as_index = True).value_counts()
    cut_IV=((grouped.unstack().iloc[:,1]/train["好坏客户"].sum()-
             grouped.unstack().iloc[:,0]/(train["好坏客户"].count()-
             train["好坏客户"].sum()))*cut_woe).sum() 
    return cut_IV
#计算各分组的IV值
cut1_IV=get_IV_data(cut1,cut1_woe)
cut2_IV=get_IV_data(cut2,cut2_woe)
cut3_IV=get_IV_data(cut3,cut3_woe)
cut4_IV=get_IV_data(cut4,cut4_woe)
cut5_IV=get_IV_data(cut5,cut5_woe)
cut6_IV=get_IV_data(cut6,cut6_woe)
cut7_IV=get_IV_data(cut7,cut7_woe)
cut8_IV=get_IV_data(cut8,cut8_woe)
cut9_IV=get_IV_data(cut9,cut9_woe)
cut10_IV=get_IV_data(cut10,cut10_woe)

IV=pd.DataFrame([cut1_IV,cut2_IV,cut3_IV,cut4_IV,cut5_IV,cut6_IV,cut7_IV,cut8_IV,cut9_IV,cut10_IV],
                index=['可用额度比值','年龄','逾期30-59天笔数','负债率','月收入','信贷数量',
                '逾期90天笔数','固定资产贷款量','逾期60-89天笔数','家属数量'],columns=['IV'])
iv=IV.plot.bar(color='b',alpha=0.3,rot=30,figsize=(10,5),fontsize=(10))
iv.set_title('特征变量与IV值分布图',fontsize=(15))
iv.set_xlabel('特征变量',fontsize=(15))
iv.set_ylabel('IV',fontsize=(15))
plt.show()

print(IV.sort_values(by='IV',ascending=False))
"""WOE轉換"""
card_new=pd.DataFrame()   #新建df_new存放woe转换后的数据
def replace_data(cut,cut_woe):
    a=[]
    cut=cut.copy()
    for i in cut.unique():
        a.append(i)
        a.sort()
    for m in range(len(a)):
        cut.replace(a[m],cut_woe.values[m],inplace=True)
    return cut

card_new["好坏客户"]=train["好坏客户"]
card_new["可用额度比值"]=replace_data(cut1,cut1_woe)
card_new["年龄"]=replace_data(cut2,cut2_woe)
card_new["逾期30-59天笔数"]=replace_data(cut3,cut3_woe)
card_new["负债率"]=replace_data(cut4,cut4_woe)
card_new["月收入"]=replace_data(cut5,cut5_woe)
card_new["信贷数量"]=replace_data(cut6,cut6_woe)
card_new["逾期90天笔数"]=replace_data(cut7,cut7_woe)
card_new["固定资产贷款量"]=replace_data(cut8,cut8_woe)
card_new["逾期60-89天笔数"]=replace_data(cut9,cut9_woe)
card_new["家属数量"]=replace_data(cut10,cut10_woe)
"""TRAIN"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
Tropfeature=['负债率','月收入','信贷数量','固定资产贷款量','家属数量']
source_X=card_new.drop(Tropfeature,axis=1)
source_X=source_X.iloc[:,1:].values
source_Y=card_new.iloc[:,:1].values
#print(source_X)
"""交叉驗證"""
X_train,X_test,y_train,y_test=train_test_split(source_X,source_Y,train_size=0.8,random_state=0)
#建立交叉验证的网格搜索进行调参
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.0001,0.001,0.01,0.1,1,10,100]}
grid=GridSearchCV(LogisticRegression(),param_grid=param_grid,cv=5)
#建立模型
coe=grid.fit(X_train,y_train)

print('最好的参数:',grid.best_params_)
print('最好的正确率{:.3f}'.format(grid.best_score_))
print('测试集的正确率{:.3f}'.format(grid.score(X_test,y_test)))
print('训练集的正确率{:.3f}'.format(grid.score(X_train,y_train)))

"""評估"""
# 导入ROC曲线模型
from sklearn.metrics import roc_curve,auc
fpr, tpr, threshold = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot( [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_curve')
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,grid.predict_proba(X_test)[:,1])
#KS
ax1=plt.plot(1 - threshold, tpr, label='tpr') # ks曲线要按照预测概率降序排列，所以需要1-threshold镜像
ax1=plt.plot(1 - threshold, fpr, label='fpr')
ax1=plt.plot(1 - threshold, tpr-fpr,label='KS')
plt.xlabel('score')
plt.title('KS Curve')
plt.ylim([0.0, 1.0])
plt.legend(loc='upper left')
plt.show()
"""建立評分卡"""
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(**grid.best_params_)
coe=model.fit(X_train,y_train)
coe=coe.coef_
coe

factor = 20 / np.log(2)
offset = 600 - 20 * np.log(20) / np.log(2)
def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores
x1 = get_score(coe[0][0], cut1_woe, factor)
x2 = get_score(coe[0][1], cut2_woe, factor)
x3 = get_score(coe[0][2], cut3_woe, factor)
x7 = get_score(coe[0][3], cut7_woe, factor)
x9 = get_score(coe[0][4], cut9_woe, factor)

print('可用额度比值对应的分数:{}'.format(x1))
print('年龄对应的分数:{}'.format(x2))
print('逾期30-59天笔数对应的分数:{}'.format(x3))
print('逾期90天笔数对应的分数:{}'.format(x7))
print('逾期60-89天笔数对应的分数:{}'.format(x9))

def compute_score(series,cut,scores):
    i=0
    list=[]
    while i<len(series):
        value=series[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j=j-1
                m=m-1
        list.append(scores[m])
        i=i+1
    return list


test1 = test

test1['x1'] = pd.Series(compute_score(test1['RevolvingUtilizationOfUnsecuredLines'], range(bins1), x1))
test1['x2'] = pd.Series(compute_score(test1['age'], range(bins2), x2))
test1['x3'] = pd.Series(compute_score(test1['NumberOfTime30-59DaysPastDueNotWorse'], bins3, x3))
test1['x7'] = pd.Series(compute_score(test1['NumberOfTimes90DaysLate'], bins7, x7))
test1['x9'] = pd.Series(compute_score(test1['NumberOfTime60-89DaysPastDueNotWorse'], bins9, x9))
test1['Score'] = test1['x1']+test1['x2']+test1['x3']+test1['x7']+test1['x9']+600

reslut=test1.loc[:,['SeriousDlqin2yrs', 'x1', 'x2', 'x3', 'x7', 'x9', 'Score']]
print(reslut)

train1['x1'] = pd.Series(compute_score(train1['RevolvingUtilizationOfUnsecuredLines'], range(bins1), x1))
train1['x2'] = pd.Series(compute_score(train1['age'], range(bins2), x2))
train1['x3'] = pd.Series(compute_score(train1['NumberOfTime30-59DaysPastDueNotWorse'], bins3, x3))
train1['x7'] = pd.Series(compute_score(train1['NumberOfTimes90DaysLate'], bins7, x7))
train1['x9'] = pd.Series(compute_score(train1['NumberOfTime60-89DaysPastDueNotWorse'], bins9, x9))
train1['Score'] = train1['x1']+train1['x2']+train1['x3']+train1['x7']+train1['x9']+600

reslut=train1.loc[:,['SeriousDlqin2yrs', 'x1', 'x2', 'x3', 'x7', 'x9', 'Score']]
print(reslut)
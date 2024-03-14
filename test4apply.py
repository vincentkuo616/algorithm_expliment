# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:55:17 2023

@author: vincentkuo
"""

import pandas as pd
from sklearn import datasets
import time
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

'''
#print(df.head())
def foo(row):
    print("The class is %s" % row["sepal length (cm)"])

time1 = time.time()
for index,row in df.iterrows():
    foo(row)
time2 = time.time()

df.apply(foo, axis = 1)

time3 = time.time()

for i in df.iloc[:,0]:
    print("123The class is %s" % i)

time4 = time.time()

print(time2-time1)
print(time3-time2)
print(time4-time3)
'''
df1 = pd.DataFrame({"a":[1,2], "b":[3,4]})
df2 = pd.DataFrame({"a":[10], "b":[20]})

print(pd.concat([df1,df2],axis=0))
print(pd.concat([df1,df2],axis=1))

df = pd.DataFrame({"a":[1,2,3], "b":[4,5,6], "c":[7,8,9], "d":[True,True,False]})
mask = df.isin({"a":[1,3], "b":[4,5], "c":[7,8,9], "d":[True,False]})
print(df[mask])

import numpy as np
a = np.nan
b = np.nan
print(a==b) # np.nan 對於任何比較的值，都會視為不相等
print(a is b)
print(np.isnan(a), np.isnan(b))

df = pd.DataFrame({"Name": ["Alice", "Bob", "Bob", "Bob", "Carol"],
                   "Score": [100, 95, 97, 97, 95]})
# 預設keep="first"，可以改成"last"來保留重複的最後一筆資料
print(df.drop_duplicates(subset=["Name"],keep="last"))

print(df["Score"].map(lambda x: "Perfect" if x==100 else "Good"))

# sort()是直接在原list排序(所以沒有回傳)
A = [1,3,5,1,2]
B = sorted(A)
print("A:", A)
C = A.sort()
print("B:", B)
print("C:", C)

#依照多個欄位進行排序且一個升序一個降序
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#print(df.sort_values(by=["sepal length (cm)","sepal width (cm)"], ascending=[True, False]))
'''
from collections import Counter
A = ["Alice", "Bob", "Bob", "Bob", "Carol"]
count_A = Counter(A)
print("Bob:", count_A["Bob"])
print("Amy:", count_A["Amy"]) # 不存在的key會回傳0而非跳error

df_car = pd.read_csv("car.csv")

#print(df_car.head(5))
count_class = Counter(df_car["class"])
print(count_class)
df2 = pd.DataFrame.from_dict(count_class,orient="index",columns=["Count"])
print(df2)
df_car_counter = pd.DataFrame({"class":list(count_class.keys()),
                               "count":list(count_class.values())})
print(df_car_counter)
# cross table
print(df_car.groupby(["class", "safety"]).size().reset_index(name="次數"))
print(df_car.groupby(["class"]).size().reset_index(name="次數").sort_values(by="次數", ascending=False)) # 嘗試 pandas 排序
print(df_car.groupby(["class"]).size().reset_index(name="次數").sort_values(by="次數", ascending=False).reset_index(drop=True).reset_index().reset_index()) # 嘗試 pandas 重設index

#格式化字串 @重要
print("補零 %02d" % 2)
print("%f四捨五入後是%.2f" % (1.235, 1.235)) # 有多個地方需要取代時，需要用()
print("熱愛{temperature}度的你".format(temperature=105)) # 用{}包一個名稱，format後面給這個名稱的值
print("{a}四捨五入後是{a:.2f}".format(a=1.235)) # 可以和%一樣指定格式
#新方法-fstring
temperature = 105
a = 1.235
print(f"熱愛{temperature}度的你") # 字串前面加f，中間用{}包變數名稱
print(f"{a}四捨五入後是{a:.2f}") # 可以和%一樣指定格式
'''
'''
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
#import nltk
#stemmer  ex. amused、amusing -> amus
ps = PorterStemmer()
#Lemmatizer  ex. amused、amusing -> amuse
lm = WordNetLemmatizer()
print(f'Stemming amusing : {format(ps.stem("amusing"))}')
print(f'lemmatization amusing : {format(lm.lemmatize("amusing", pos = "v"))}')
'''

text = """
中央流行疫情指揮中心今(29)日公布國內新增11例COVID-19確定病例，分別為1例本土及10例境外移入；另確診個案中無新增死亡。

指揮中心表示，今日新增1例本土個案(案16326)，為印尼籍30多歲女性，今(2021)年9月27日出現頭痛症狀，9月28日就醫採檢，於今日確診。衛生單位已匡列接觸者7人，均列居家隔離，其餘接觸者匡列中。

指揮中心指出，今日新增10例境外移入個案，為9例男性、1例女性，年齡介於20多歲至40多歲，入境日介於9月3日至9月28日，分別自美國(案16316)、哈薩克(2例，案16317、案16318)、巴基斯坦(案16319)、柬埔寨(16320)、俄羅斯(案16324)及菲律賓(案16325)入境，餘3例 (案16321、案16322、案16323)的旅遊國家調查中；詳如新聞稿附件。

指揮中心統計，截至目前國內累計3,358,228例新型冠狀病毒肺炎相關通報(含3,341,439例排除)，其中16,216例確診，分別為1,581例境外移入，14,581例本土病例，36例敦睦艦隊、3例航空器感染、1例不明及14例調查中；另累計110例移除為空號。2020年起累計842例COVID-19死亡病例，其中830例本土，個案居住縣市分布為新北市412例、臺北市318例、基隆市28例、桃園市26例、彰化縣15例、新竹縣13例、臺中市5例、苗栗縣3例、宜蘭縣及花蓮縣各2例，臺東縣、雲林縣、臺南市、南投縣、高雄市及屏東縣各1例；另12例為境外移入。

指揮中心再次呼籲，民眾應落實手部衛生、咳嗽禮節及佩戴口罩等個人防護措施，減少不必要移動、活動或集會，避免出入人多擁擠的場所，或高感染傳播風險場域，並主動積極配合各項防疫措施，共同嚴守社區防線。
"""
import re
print(re.findall("新增\d+例境外", text))
print(re.findall("新增(\d+)例境外", text))
print(re.findall("..[縣市]", text)) # []內表示都可以

import matplotlib.pyplot as plt
'''
from pandas.plotting import parallel_coordinates
import seaborn as sns
iris = sns.load_dataset('iris') # 換個資料範例用iris
parallel_coordinates(iris, 'species')
plt.show()
'''
import twstock
stockid = "2538"
stock = twstock.Stock(stockid) #0050 2330 2603 2347 2449 1440 2538
price = np.array(stock.price)
date = stock.date
#plt.plot(date,price)
#plt.xticks(rotation = 30)
#plt.title("0050")

# 尋找峰值
from scipy.signal import argrelextrema
max_index = argrelextrema(price, np.greater)[0]
peak = price[max_index]  
#print(f"peak:{peak}")
# 繪圖峰值
plt.plot(date,price)
plt.xticks(rotation = 30)
for index in max_index:
    plt.scatter(date[index],price[index],c="r")
plt.show()
plt.clf() # 清除圖片暫存

# 計算移動平均
series_price = pd.Series(price)
series_price_ma3 = series_price.rolling(3).mean()
series_price_ma5 = series_price.rolling(5).mean()
# 移動平均繪圖比較
p1 = plt.plot(date,series_price)
p2 = plt.plot(date,series_price_ma3)
p3 = plt.plot(date,series_price_ma5,c="r")
plt.legend(['origin', '3 day MA', "5 day MA"])
plt.xticks(rotation = 30)
plt.title(stockid)

#import logging
#logging.basicConfig(level=logging.DEBUG, format="[% (levelname)s] %(asctime)s - %(message)s")
#logging.info("step 1")
import warnings
warnings.filterwarnings("ignore")

# enumerate 枚舉
# using own counter variable
lst = [1, 2, 3]
i = 0
for x in lst:
    print(i)
    i += 1

# using enumerate
i = 0
for i, val in enumerate(lst):
    print(i, val)
    pass


# default mutable arguments
def append(n, l=[]):
    l.append(n)
    return l
    
# setting default to None
def append2(n, l=None):
    if l is not None:
        l.append(n)
        return l

## o/p
l1 = append2(0)   # [0]
l2 = append2(1)   # [0, 1]

# looping through items
d = {"val1": 1, "val2": 2, "val3": 3}
for key, val in d.items():
    print(key, val)
    pass

from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)

if type(p) == tuple:
    print('1',p)
if isinstance(p, tuple):
    print('2',p)

def checking_performance1():
    start_time = time.time()
    time.sleep(1)
    end_time = time.time()
    print(end_time - start_time)

from time import perf_counter
def checking_performance2():
    start_time = perf_counter()
    time.sleep(1)
    end_time = perf_counter()
    print(end_time - start_time)

checking_performance1()
checking_performance2()

t=[]
s=["H","e","l","l","o"]
s.reverse()
print(s)

#import json
#import pandas as pd
#from urllib import request
##氣象局-鄉鎮天氣預報-台灣未來1週天氣預報
##https://opendata.cwb.gov.tw/dataset/statisticDays/F-D0047-091
#url = 'https://opendata.cwb.gov.tw/fileapi/v1/opendataapi/F-D0047-091?Authorization=CWB-413ECD83-6DED-44E7-AEE1-0A75A6394562&downloadType=WEB&format=JSON'
#data = request.urlopen(url).read().decode("utf-8")
#print (json.loads(data))

#import requests
#url='https://goodinfo.tw/StockInfo/StockBzPerformance.asp?STOCK_ID=2347' #4919
#headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'
#} # 假的HEADER
#res = requests.get(url, headers = headers)
#res.encoding ='utf-8'
#from bs4 import BeautifulSoup
##解析器：lxml(官方推薦，速度最快)
#soup = BeautifulSoup(res.text, 'lxml') 
#data = soup.select_one('#txtFinDetailData')
#import pandas
#dfs = pandas.read_html(data.prettify())
#print(len(dfs))
#df = dfs[0]
#print(df.head())


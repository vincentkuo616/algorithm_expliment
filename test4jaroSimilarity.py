# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:12:46 2024

@author: vincentkuo
"""

from math import floor, ceil

# Function to calculate the Jaro Similarity of two strings

def jaro_distance(s1, s2):
    
    # If the s are equal
    if (s1==s2):
        return 1.0
    
    # Length of two s
    len1 = len(s1)
    len2 = len(s2)
    
    # Maximum distance upto which matching is allowed
    max_dist = floor(max(len1, len2) / 2) - 1
    
    # Count of matches
    match = 0
 
    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first
    for i in range(len1):
 
        # Check if there is any matches
        for j in range(max(0, i - max_dist), 
                       min(len2, i + max_dist + 1)):
             
            # If there is a match
            if (s1[i] == s2[j] and hash_s2[j] == 0):
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break
 
    # If there is no match
    if (match == 0):
        return 0.0
 
    # Number of transpositions
    t = 0
    point = 0
 
    # Count number of occurrences
    # where two characters match but
    # there is a third matched character
    # in between the indices
    print(hash_s1,hash_s2)
    for i in range(len1):
        if (hash_s1[i]):
 
            # Find the next matched character
            # in second
            while (hash_s2[point] == 0):
                point += 1
 
            if (s1[i] != s2[point]):
                t += 1
            point += 1
    t = t//2
 
    # Return the Jaro Similarity
    print(match,t)
    return (match/ len1 + match / len2 +
            (match - t) / match)/ 3.0

# Jaro Winkler Similarity 
def jaro_Winkler(s1, s2) : 
 
    jaro_dist = jaro_distance(s1, s2); 
 
    # If the jaro Similarity is above a threshold 
    if (jaro_dist > 0.7) :
 
        # Find the length of common prefix 
        prefix = 0; 
 
        for i in range(min(len(s1), len(s2))) :
         
            # If the characters match 
            if (s1[i] == s2[i]) :
                prefix += 1; 
 
            # Else break 
            else :
                break; 
 
        # Maximum of 4 characters are allowed in prefix 
        prefix = min(4, prefix); 
 
        # Calculate jaro winkler Similarity 
        jaro_dist += 0.1 * prefix * (1 - jaro_dist); 
 
    return jaro_dist

# Driver code
s1 = "CRATE"
s2 = "TRACE"
s1 = "CRATE"
s2 = "RCATE"
s1 = "Johnny"
s2 = "Johny"
s1 = "arnab"
s2 = "raanb"
s1 = "arnab"
s2 = "ranba"
#s2 = "banra"
#s1 = "apple"
#s2 = "ppabe"
#s1 = "fox"
#s2 = "afoxbcd"
#s2 = "foxbcd"
s1 = "FAREMVIEL"
s2 = "FARMVILLE"
s1 = "data of engineer"
s2 = "date of engineer"
s2 = "engineer data"
s1 = "the necklace"
s2 = "he need lack"
s1 = "the perfect"
s2 = "her prefer"
s1 = "CLI-751Y"
s2 = "CLI-726C"
s2 = "CLI-771XL M"
 
# Prjaro Similarity of two s
print(round(jaro_distance(s1, s2),6))
print(round(jaro_Winkler(s1, s2),6))
import Levenshtein
print(Levenshtein.distance(s1, s2))
print(Levenshtein.ratio(s1, s2))
print(Levenshtein.jaro(s1, s2))

from fuzzywuzzy import fuzz
print(fuzz.ratio(s1, s2))
print(fuzz.partial_ratio(s1, s2)) # 比較部分字串相似度
print(fuzz.token_sort_ratio(s1, s2)) # 忽略詞序
print(fuzz.token_set_ratio(s1, s2)) # 忽略重複的單字

print("------Hamming Distance--------------------------------")
import textdistance as td
'''Hamming Distance - 多少位置具有不同字元
'''
print(td.hamming('bellow', 'below'))
print(td.hamming.normalized_similarity('bellow', 'below'))

print("------Levenshtein Distance--------------------------------")
'''Levenshtein Distance - 編輯距離 (插入、刪除、替換)
'''
print(td.levenshtein('bellow', 'below'))
print(td.levenshtein.normalized_similarity('bellow', 'below'))

print("------Damerau-Levenshtein Distance--------------------------------")
'''Damerau-Levenshtein Distance - 編輯距離 (插入、刪除、替換、轉置)
'''
print(td.levenshtein('act', 'cat'))
print(td.levenshtein.normalized_similarity('act', 'cat'))
print(td.damerau_levenshtein('act', 'cat'))
print(td.damerau_levenshtein.normalized_similarity('act', 'cat'))
print(td.damerau_levenshtein('bellow', 'below'))
print(td.damerau_levenshtein.normalized_similarity('bellow', 'below'))

print("------Jaro Similarity--------------------------------")
'''Jaro Similarity - 基於匹配字元數量和換位的演算法
'''
print(td.jaro('bellow', 'below'))

print("------Jaro-Winkler Similarity--------------------------------")
'''Jaro-Winkler Similarity - 基於匹配字元數量和換位的演算法
'''
print(td.jaro_winkler('bellow', 'below'))

print("------Smith-Waterman Similarity--------------------------------")
'''Smith-Waterman Similarity - 尋找兩個序列之間的局部最佳比對 與 Needleman-Wunsch 全域最佳比對不同
                               該演算法在生物資訊學中特別有用，用於鑑別生物序列DNA RNA 的相似度
'''
print(td.smith_waterman('bellow', 'below'))
print(td.smith_waterman.normalized_distance('bellow', 'below'))
print(td.smith_waterman('GCATGCG', 'GATTACA'))
print(td.smith_waterman.normalized_distance('GCATGCG', 'GATTACA'))

print("------Needleman-Wunsch Similarity--------------------------------")
'''Needleman-Wunsch Similarity - 尋找兩個序列之間的全域最佳比對 對齊兩個完整序列
'''
print(td.needleman_wunsch('bellow', 'below'))
print(td.needleman_wunsch.normalized_distance('bellow', 'below'))
print(td.needleman_wunsch('GCATGCG', 'GATTACA'))
print(td.needleman_wunsch.normalized_distance('GCATGCG', 'GATTACA'))

print("------Jaccard Similarity--------------------------------")
'''Jaccard Similarity - 尋找兩個集合之間的相似性 交集/聯集(A+B-交集)
'''
print(td.jaccard('bellow', 'below'))
print(td.jaccard.normalized_distance('bellow', 'below'))
print(td.jaccard('Jaccard Similarity'.split(), 'Similarity Similarity Jaccard'.split()))
print(td.jaccard.normalized_distance('Jaccard Similarity'.split(), 'Similarity Similarity Jaccard'.split()))


print("------Sorensen-Dice Similarity--------------------------------")
'''Sorensen-Dice Similarity - 尋找兩個集合之間的相似性 給交集更高的權重 2交集/A+B
'''
print(td.sorensen('bellow', 'below'))
print(td.sorensen.normalized_distance('bellow', 'below'))
print(td.sorensen('Jaccard Similarity'.split(), 'Similarity Similarity Jaccard'.split()))
print(td.sorensen.normalized_distance('Jaccard Similarity'.split(), 'Similarity Similarity Jaccard'.split()))

print("------Tversky Index--------------------------------")
'''Tversky Index - 尋找兩個集合之間的相似性 (當處理不平衡資料 或 集合中元素的存在、不存在具有不同重要性[權重]的情況，特別有用)
'''
tversky = td.Tversky(ks=(1, 1))
print(tversky('Jaccard Similarity'.split(), 'Similarity Similarity Jaccard'.split())) # Same as Jaccard
tversky = td.Tversky(ks=(0.5, 0.5))
print(tversky('Jaccard Similarity'.split(), 'Similarity Similarity Jaccard'.split())) # Same as Sorensen-Dice
tversky = td.Tversky(ks=(0.2, 0.8))
print(tversky('Jaccard Similarity'.split(), 'Similarity Similarity Jaccard'.split()))
print(tversky('Jaccard Similarity', 'Similarity Similarity Jaccard')) # 18 / (18+0.2*0+0.8*11)

print("------Overlap Similarity--------------------------------")
'''Overlap Similarity - 重疊相似度 即重疊比率，不管順序  X∩Y / min(X,Y)
'''
print(td.overlap('bellow', 'below'))
print(td.overlap('bellow', 'wolley'))

print("------Cosine Similarity--------------------------------")
'''Cosine Similarity - 餘弦相似度 作用在比較過程中標準化文件長度，介於0~1之間
'''
print(td.cosine('bellow', 'below')) # 5 / sqrt(5*6)
print(td.cosine('bellow', 'wolley')) # 5 / sqrt(6*6)

print("------N-gram (Trigrams)--------------------------------")
'''N-gram (Trigrams) - 擷取n個字元的重疊序列 計算相似度
'''
def trigrams(a,b):
    l_a, l_b = len(a), len(b)
    if l_a==0 or l_b==0:    return 0
    ab = []
    for i in a:
        if i not in ab:
            ab.append(i)
    for j in b:
        if j not in ab:
            ab.append(j)
    l_ab = len(ab)
    return (l_a+l_b-l_ab)/l_ab
    
print(trigrams(td.find_ngrams('trigrams',3),td.find_ngrams('trigrasm',3))) # (6+6-8)/8
print(trigrams(td.find_ngrams('bellow',3),td.find_ngrams('below',3))) # (4+3-5)/5
print(trigrams(td.find_ngrams('bellow',3),td.find_ngrams('wolley',3)))

print("------Ratcliff-Obershelp--------------------------------")
'''Ratcliff-Obershelp - 使用LCS (尋找最長公共子字串) 迭帶
'''
print(td.ratcliff_obershelp('RO practice', 'RO pattern matching'))
print(td.ratcliff_obershelp('RO pattern matching', 'RO practice'))
print(td.lcsstr('RO practice', 'RO pattern matching'))
print(td.lcsseq('RO practice', 'RO pattern matching'))
print(td.lcsseq('RO pattern matching', 'RO practice'))

#'''
import pandas as pd

data = pd.read_excel("C:\\Users\\vincentkuo\\Downloads\\testData4Jaro.xlsx")
#print(data.head())
dataGroup = data.groupby("PS")
dataGroupI = dataGroup.get_group("ICCAN")

#print(dataGroupI["料號"].values)
test1=dataGroupI["料號"].values
test2=dataGroupI["料號"].values
output=[]
output_top3=[]
for i in test1:
    temp=[]
    temp_top3=[]
    for j in test2:
        if i==j:
            temp.append(1)
        else:
#            temp.append(round(td.levenshtein.normalized_similarity(i,j),4))
            temp.append(round(td.jaro_winkler(i,j),4))
    for l in range(len(test2)):
        if i==test2[l]:
            temp_top3.append([l,1])
        else:
            temp_top3.append([l,round(td.jaro_winkler(i,test2[l]),4)])
    output.append(temp)
    temp_top3_sort = sorted(temp_top3, key = lambda i:i[1], reverse=True)[1:4]
    temp_top3_sort = [i for item in temp_top3_sort for i in item]
    output_top3.append(temp_top3_sort)
            
print(output)


temp_all=[['ITEM A','ITEM B','Jaro_Winkler','Jaro','Hamming','Levenshtein','Damerau-Levenshtein']]
for key in dataGroup.groups.keys():
    dataGroupI = dataGroup.get_group(key)
    test1=dataGroupI["料號"].values
    test2=dataGroupI["料號"].values

    for i in range(len(test1)):
        for j in range(i):
            temp_all.append([test1[i],test2[j],round(td.jaro_winkler(test1[i],test2[j]),4),round(td.jaro(test1[i],test2[j]),4),
                             round(td.hamming.normalized_similarity(test1[i],test2[j]),4),round(td.levenshtein.normalized_similarity(test1[i],test2[j]),4),
                             round(td.damerau_levenshtein.normalized_similarity(test1[i],test2[j]),4)])
    


#outputDF = pd.DataFrame(output)
outputDF2 = pd.DataFrame(output)
outputDF3 = pd.DataFrame(output_top3)
#'''




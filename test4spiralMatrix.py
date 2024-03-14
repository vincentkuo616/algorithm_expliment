# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:57:05 2023

@author: vincentkuo
"""

matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
#matrix = [[1,2,3],[4,5,6],[7,8,9]]
matrix = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
matrix = [[1,2,3]]
matrix = [[1],[2],[3]]
'''
 1  2  3  4         1 2 3        1  2  3        1  2  3  4
 5  6  7  8         4 5 6        4  5  6        5  6  7  8
 9 10 11 12         7 8 9        7  8  9        9 10 11 12
13 14 15 16                     10 11 12
'''
res=[]
left,right=0,len(matrix[0])
top,bottom=0,len(matrix)

while left<right and top<bottom:
    for i in range(left,right):
        res.append(matrix[top][i])
    top+=1
    
    for i in range(top,bottom):
        res.append(matrix[i][right-1])
    right-=1

    if not(left<right and top<bottom):
        break
    for i in range(right-1,left-1,-1):
        res.append(matrix[bottom-1][i])
    bottom-=1

    for i in range(bottom-1,top-1,-1):
        res.append(matrix[i][left])
    left+=1
print(res)

tes=[]
left,right=0,len(matrix[0])
top,bottom=0,len(matrix)
while left<right and top<bottom:
    for i in range(left, right):
        tes.append(matrix[top][i])
    top+=1
    
    for i in range(top,bottom):
        tes.append(matrix[i][right-1])
    right-=1
    
    if top==bottom or left==right:
        break
#    print(top, bottom, left, right)
    for i in range(right-1,left-1,-1):
        tes.append(matrix[bottom-1][i])
    bottom-=1
    
    for i in range(bottom-1,top-1,-1):
        tes.append(matrix[i][left])
    left+=1

print(tes)
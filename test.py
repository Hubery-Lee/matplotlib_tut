'''
Description: 
version: 
Author: Hubery-Lee
E-mail: hrbeulh@126.com
Date: 2022-10-14 09:00:36
LastEditTime: 2022-10-14 16:44:26
LastEditors: Hubery-Lee
'''

import matplotlib.pyplot as plt
from  collections import OrderedDict  
import numpy as np

data = OrderedDict()

f = open('temperature.txt',encoding='UTF8')
flag =True
for line in f:
    if '2022年' in line:
        time = line.split(' ')[4].strip()
        # print(time)
        data[time]=[]
    
    if 'coretemp-isa-0001' in line:
        flag = True

    if 'coretemp-isa-0000' in line:
        flag = False
             
    if flag==True and 'Core 0:' in line:
        core0 = line.split(' ')[9].strip().replace('+','').replace('°C','')  #strip()丢弃空行等
        # print(core0)
        data[time].append(core0)
    
    if flag==True and 'Core 1:' in line:
        core1 = line.split(' ')[9].strip().replace('+','').replace('°C','')  #strip()丢弃空行等
        # print(core0)
        data[time].append(core1)
        
print(data)

x = []
y = []

for time in data:
    x.append(time)
    y.append(data[time][0])

print(x)
print(y)

figure = plt.figure(figsize=(18,5))
plt.plot(x,y)
plt.xlabel('time')
plt.ylabel('temperature')
plt.savefig('temperature.png')
# plt.show()
    

    
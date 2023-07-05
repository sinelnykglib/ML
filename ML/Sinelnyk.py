import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

m = 9
n = m * 300
'''
1.  Відкрити та зчитати файл з даними.
'''
weather = pd.read_csv('Weather.csv')

'''
2.  Визначити та вивести кількість записів та кількість полів у кожному записі
'''
#print(weather)
'''
3.  Вивести 5 записів, починаючи з М-ого (число М – місяць народження
    студента, має бути визначено як змінна), та кожен N-ий запис, де число
    N визначається як 500 * М для місяця з першого півріччя та 300 * М для
    місяця з другого півріччя.
'''
#print(f"\nK = {m} \n", weather[m:m+5], "\n\n", weather[n::n])

'''
4. Визначити та вивести тип полів кожного запису.
'''
#print(weather.dtypes)

'''
5.  Замість поля СЕТ ввести нові текстові поля, що відповідають числу,
    місяцю та року. Місяць та число повинні бути записані у
    двоцифровому форматі.
'''
weather['year'] = weather['CET'].apply(lambda x: x.split('-')[0])
weather['month'] = weather['CET'].apply(lambda x: x.split('-')[1])
weather['day'] = weather['CET'].apply(lambda x: x.split('-')[2])
weather['month'] = weather['month'].apply(lambda x: x.zfill(2))
weather['day'] = weather['day'].apply(lambda x: x.zfill(2))
weather = weather.drop(columns='CET')
#print(weather)

'''
6.   Визначити та вивести:
        a. Кількість днів із порожнім значенням поля Events;
        b. День, у який середня вологість була мінімальною, а також
        швидкості вітру в цей день;
        c. Місяці, коли середня температура від нуля до п’яти градусів..
'''
# a
#count = weather[' Events'].isnull().sum()
#print('Кількість днів із порожніми значеннями поля Events',count)

# b

#print(weather[weather[' Mean Humidity'] == weather[' Mean Humidity'].min()][['year','month','day', ' Mean Humidity', ' Max Wind SpeedKm/h', ' Mean Wind SpeedKm/h',' Max Gust SpeedKm/h']].to_string())

#c +-

#temp = weather[(weather['Mean TemperatureC'] >= 0) & (weather['Mean TemperatureC'] <= 5)]
#print(temp[['month','Mean TemperatureC']])

'''
7. Визначити та вивести:
    a. Середню максимальну температуру по кожному дню за всі роки;
    b. Кількість днів у кожному році з туманом.
'''

#a

#group_data = weather.groupby(['month','day'])['Max TemperatureC'].mean()

#print(group_data)

#b

#weather['Fog'] = weather[' Events'].str.contains('Fog')

#grop_dat = weather.groupby(['year','Fog'],as_index=False)[' Events'].count()

#print(grop_dat[grop_dat['Fog']][['year',' Events']])

'''
8. Побудувати стовпчикову діаграму кількості Events
'''
#bard = weather.groupby([' Events'])['day'].count()
#plt.figure()
#bard.plot.bar()
#plt.show()

'''
9.  Побудувати кругову діаграму напрямків вітру (сектор на діаграмі має
    відповідати одному з восьми напрямків – північний, південний, східний,
    західний та проміжні).
'''
wind_count = weather.groupby(['WindDirDegrees'],as_index=False)['day'].count()
way = ['north','northeast','east','southeast','south','southwest','west','northwest']
food = [wind_count[(wind_count['WindDirDegrees']<=23)|(wind_count['WindDirDegrees']>=338)]['day'].sum()]
for i in range(1,8):
    food.append(wind_count[(wind_count['WindDirDegrees']<=(i*45+23))&(wind_count['WindDirDegrees']>=(i*45-22))]['day'].sum())
print(food)
plt.pie(food,labels=way)
plt.show()

'''
10. Побудувати на одному графіку (тип графіка обрати самостійно!):
        a. Середню по кожному місяцю кожного року максимальну
         температуру;
        b. Середню по кожному місяцю кожного року мінімальну точку
         роси.
'''

#a

#group_data = weather.groupby(['month','year'])['Max TemperatureC'].mean()
#group_data.plot.area()

#b

#gr_dt = weather.groupby(['month','year'])['Min DewpointC'].mean()
#gr_dt.plot()
#plt.ylim(-12,40)
#plt.legend(('avg_temp','avg_Min_Dewpoint'))
#plt.show()
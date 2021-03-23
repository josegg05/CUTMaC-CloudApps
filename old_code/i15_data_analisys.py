#%%
import pandas as pd
import numpy as np
import datetime
pd.set_option("precision", 3)

#%%
# Data Load
data_file_name = "datasets/la_vegas/i15_bugatti/bugatti_data_lean.csv"
# data_file_name = "datasets/la_vegas/i15_bugatti/bugatti_data.csv"

data = pd.read_csv(data_file_name)
# data = data.drop(columns=['Path', 'RoadIndex', 'RoadwayID', 'SegmentID', 'DeviceID', 'Volume1', 'Volume2', 'Volume3',
#                           'Volume4', 'Volume5', 'Volume6','RoadType', 'Location', 'Polling_Period', 'DayOfWeek',
#                           'DateValue', 'HourIdx', 'Holiday'])
# data.to_csv('bugatti_data_lean.csv', index=False)

#%%
# # Forma 1 --> Solo para convertir un valor
# data_time = data['DateTimeStamp']
# print(type(data_time.iloc[0]))
# print(data_time.iloc[0])
# date_time_str = data_time.iloc[0]
# date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
#
# print('Date:', date_time_obj.date())
# print('Time:', date_time_obj.time())
# print('Date-time:', date_time_obj)

#%%
# Forma 2 --> convertir la columna completa
data['DateTimeStamp'] = pd.to_datetime(data['DateTimeStamp'])
data = data.sort_values(by=['DateTimeStamp','DetectorID'],ascending=[True, True])

date_time_obj = data['DateTimeStamp'].iloc[0]
print('Date:', date_time_obj.date())
print('Time:', date_time_obj.time())
print('Minute:', date_time_obj.time().minute)
print('Date-time:', date_time_obj)

print(data['DateTimeStamp'].unique())
print(len(data['DateTimeStamp'].unique()))

#%%
date_unq = pd.Series(data['DateTimeStamp'].unique())
date_rest = []
date_rare = []
date_no_15 = []
for i in range(len(date_unq)-1):
    if date_unq[i+1].time().hour == date_unq[i].time().hour:
        rest = date_unq[i+1].time().minute - date_unq[i].time().minute
    elif (date_unq[i+1].time().hour > date_unq[i].time().hour) or ((date_unq[i+1].time().hour == 0) and (date_unq[i].time().hour == 23)):
        rest = date_unq[i+1].time().minute + 60 - date_unq[i].time().minute
    else:
        rest = -1
        date_rare.append([date_unq[i+1], date_unq[i]])
        print("algo raro")
    date_rest.append(rest)
    if rest != 15:
        date_no_15.append([rest, date_unq[i+1], date_unq[i]])

#print(np.unique(date_rest))
print(date_rare)
print('\n\n Distintos de 15 y 16:')
print(date_no_15)
print(len(date_no_15))
#%%
num = list(np.unique(date_rest))
count = np.zeros(len(num))
for i in date_no_15:
    if i[0] in num:
        count[num.index(i[0])] += 1
#print(count)

fusion = []
for i in range(len(num)):
    fusion.append([num[i], count[i]])
print(fusion)

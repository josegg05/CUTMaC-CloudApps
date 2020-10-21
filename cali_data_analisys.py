import pandas as pd
import numpy as np
import csv


filename = "datasets/california_paper_eRCNN/I5-N-3/2015_2.csv"
print('\tParsing', filename)

timesteps = set()
sections = set()
data = []

with open(filename, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',', quotechar='\"')
    for row in reader:
        timesteps.add(int(row[0]))
        sections.add(int(row[1]))
        data.append(row[2:])

data = np.asarray(data, dtype=np.float32)
num_sections = max(sections) + 1
num_timesteps = max(timesteps) + 1

print(timesteps)
print(sections)
print(data)
print(num_sections)
print(num_timesteps)

sequence = []
for i in range(num_timesteps):
    stack = None
    for j in range(num_sections):
        stack = np.vstack([stack, data[i * num_sections + j]]) if stack is not None else data[i * num_sections + j]
    sequence.append(stack)

print(sequence)

#%%

time_window = 72
time_aggregation = 1
forecast_window = 1
forecast_aggregation = 1
max_timestep = num_timesteps - time_window * time_aggregation - forecast_window * forecast_aggregation + 1
print(max_timestep)
d = []
labels = []
for i in range(0, 2, time_aggregation):
    time_steps = []
    for j in range(time_window):
        initial_index = i + j * time_aggregation
        final_index = i + (j + 1) * time_aggregation
        time_steps.append(np.mean(np.stack(sequence[initial_index:final_index], axis=1), axis=1))
    d.append(np.stack(time_steps, axis=1))
    print(f"La imagen {i} es : {d[i]}")
    forecast_steps = []
    for j in range(forecast_window):
        initial_index = i + time_window + j * forecast_aggregation
        final_index = i + time_window + (j + 1) * forecast_aggregation
        print(f"initial index: {initial_index}")
        print(f"final index: {final_index}")
        print(np.mean(np.stack(sequence[initial_index:final_index], axis=1), axis=1))
        forecast_steps.append(np.mean(np.stack(sequence[initial_index:final_index], axis=1), axis=1))
    labels.append(np.stack(forecast_steps, axis=1))
    print(f"El label {i} es : {labels[i]}")

#%%
print(len(labels))

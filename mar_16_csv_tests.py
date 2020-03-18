import csv
import numpy as np
import ast

A = [[5,1,9],[0.8,94,0.9],[0,2,6]]
d = {"hey":62,"un":0.88}

with open('mar_16.csv', 'w+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([str(d)])
    writer.writerow(["col1","col2","mean"])

    writer.writerows(A)

print(np.loadtxt('mar_16.csv', delimiter = ",", skiprows = 2))


with open('mar_16.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        h = ast.literal_eval(*row)
        break

print(h)
print(type(h))

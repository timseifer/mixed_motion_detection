# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

infile = '/Users/timseifert/multi-object-tracker/examples/example_notebooks/bounding_box_collection_first_ten.csv'



def plotLine(p1, p2):
    plt.plot( [p1[0], p2[0]], [p1[1], p2[1]] )

COLS = ['id','x','y']
df = pd.DataFrame(columns=COLS)


x =0
x2=0
y=0
y2=0
x_final=[]
y_final = []
id_point =[]


with open(infile, 'r') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        try:
            coords = row[1]
            coords = coords.strip("[]")
            arr = coords.split(" ")
            while '' in arr:
                arr.remove('')
            x = float(arr[0])
            x2 = float(arr[1])
            y = (float(arr[2]))
            y2 = float(arr[3])

            x_final.append(((x+y2)))
            #we must invert y coordinates
            y_final.append(((y+x2)))
            # print("x is" + str(float(x)))
            # print("y is" + str(float(y)))
            id_point.append(row[0])
            # take coordinates and output to a csv file
            new_entry = []
            new_entry.append(row[0])
            new_entry.append((x+x2))
            new_entry.append((y+y2))
        
            data_frame = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(data_frame, ignore_index=True)  
            df.to_csv('id_and_coordinates.csv', columns=COLS,index=False) 
        except:
            continue
     
# Fill a dictionary with the points
points = dict()# Create empty dictionary
for idx, point in enumerate(zip(x_final,y_final)):
    if id_point[idx] in points:
        # append the new number to the existing array at this slot
        points[id_point[idx]].append(list(point))
    else:
        # create a new array in this slot
        points[id_point[idx]] = list(point)


#'/Users/timseifert/multi-object-tracker/examples/assets/ppl_walking.png'
#
im = plt.imread('/Users/timseifert/multi-object-tracker/examples/assets/ppl_walking.png')
plt.figure(figsize=(10,10))
implot = plt.imshow(im)
plt.scatter(x_final, y_final)

delay =''
curr = ''
for key in points:
		for pnt in points[key]:
			delay = curr
			curr = pnt
			try:
				plt.annotate(key, xy=pnt,size=5)
			except:
				continue
			try:
				plotLine(curr, delay)
			except:
				continue
plt.gca()
plt.show()

# %% [markdown]
# What's happending here is we plot the bottom left of the bounding box


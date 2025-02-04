import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

point_cloud = pd.read_csv('example_data.csv')
point_cloud = point_cloud[['Points_m_XYZ_0', 'Points_m_XYZ_1', 'Points_m_XYZ_2']]
point_cloud.rename(columns={'Points_m_XYZ_0': 'x', 'Points_m_XYZ_1': 'y', 'Points_m_XYZ_2': 'z'}, inplace=True)
labels = np.random.choice(['mound', 'ground'], size=len(point_cloud))
point_cloud['label'] = labels
train_df, test_df = train_test_split(point_cloud, test_size=0.2, random_state=42)

train_df.to_csv('point_cloud_train.csv')
test_df.to_csv('point_cloud_test.csv')

def csv_to_off(csv_file, off_file):
    with open(csv_file, 'r') as csv_input:
        reader = csv.reader(csv_input)
        next(reader)  # Skip header if exists
        
        # Read data from CSV
        vertices = []
        for row in reader:
            vertices.append([float(row[1]), float(row[2]), float(row[3]), (row[4])])

    # Write data to OFF file
    with open(off_file, 'w') as off_output:
        off_output.write("OFF\n")
        off_output.write("{} 0 0\n".format(len(vertices)))
        for vertex in vertices:
            off_output.write("{} {} {} {}\n".format(vertex[0], vertex[1], vertex[2], vertex[3]))

# Example usage:
csv_to_off('point_cloud_train.csv', "train.off")
csv_to_off('point_cloud_test.csv', "test.off")

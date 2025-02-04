import numpy as np
import subprocess
import os

# converts input PCAP to PCD files
def pcap_to_pcd(input_file, output_folder):
    subprocess.run(["python", "main.py", "-p", input_file, "-o", output_folder, "-c", "params.yaml"])
    print('dexgtch')
    os.chdir(output_folder)

output_folder = "/PointNetTest/pcd_output"
datadir = output_folder + "/velodynevlp16/data_pcl"
input_file = "/PointNetTest/HDL32-V2_Monterey Highway.pcap"
pcap_to_pcd(input_file, output_folder)

def rename_files(datadir):
    # Iterate over all files in the folder
    for filename in os.listdir(datadir):
        # Split the filename by underscore
        parts = filename.split('_')
        if len(parts) > 1:
            # Get the characters before the first underscore
            new_filename = parts[0] + os.path.splitext(filename)[1]
            # Rename the file
            os.rename(os.path.join(datadir, filename), os.path.join(datadir, new_filename))
            print(f"Renamed '{filename}' to '{new_filename}'")
        else:
            print(f"No underscore found in '{filename}', skipping...")
import struct
import os

def load_data(datadir, num_frames):
    '''Obtains coordinate points and quaternions for pose from pcd files'''
    points = []
    quaternions = []
    for i in range(num_frames):
        filename = '%s/%s.pcd' % (datadir, i)
        # Open the binary file in read mode
        with open(filename, 'rb') as fd:
            lines = fd.readlines()
            quaternion = lines[7][16:]
            num_points = int(lines[5][6:])
            unpacker = struct.Struct("fff4B")
            packed_data = lines[-1]
            p = []

            for j in range(0, num_points):
                try:
                    unpacked = unpacker.unpack_from(packed_data[j:])
                except struct.error:
                    break

                p.append((unpacked[0], unpacked[1], unpacked[2]))
            fd.close()
        points.append(p)
        quaternions.append(quaternion)
    return points, quaternions

datadir = "pcd_output/velodynevlp16/data_pcl"
points, quaternions = load_data(datadir, 1362)

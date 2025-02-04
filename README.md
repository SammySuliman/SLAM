# SLAM

This repo is an example of work I did in the summer of 2024 when I was first trying to understand SLAM (Simultaneous Localization & Mapping).

The directory EKF-SLAM is a 3D expansion of the open-source 2D Extended Kalman Filter SLAM algorithm written by Atsushi Sakai avaliable here: https://atsushisakai.github.io/PythonRobotics/modules/4_slam/ekf_slam/ekf_slam.html

Please note my 3D implementation has not been formally validated.

The directory PointNetClassifier contains my fine-tuning of the PointNet, avaliable here: https://github.com/charlesq34/pointnet \
PointNetClassifier is a binary classifier that distinguishes between LiDAR point clouds representing "ground" or undisturbed earth and "mound", representing heaped raw material.

The ideal workflow is that the 3D EKF-SLAM will be able to localize the agent using LiDAR data and extract the point clouds of interests which can be classified as being part of the "ground" or "mounds" by the PointNetClassiifer. 


! WARNING !
I developed the PointNetClassifier locally, and some associated files were too large to transfer to GitHub. For example, the .pcap file originally containing my sample LiDAR data "HDL32-V2_Monterey Highway.pcap" is not avaliable here and instead can be found here: https://data.kitware.com/#item/5b7ffd798d777f06857cb530. However, the file was already converted to .pcd form for training through the file pcap_to_pcd.py, so it is not necessary to access it.

If there are any other issues experienced while attempting to run this model, please open a new issue and I will try to help.

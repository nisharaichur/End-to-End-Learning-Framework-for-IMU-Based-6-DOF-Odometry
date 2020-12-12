# Implements a paper: End-to-End-Learning-Framework-for-IMU-Based-6-DOF-Odometry 
https://www.mdpi.com/1424-8220/19/17/3777/htm

The proposed inertial odometry method allows leveraging inertial sensors that are widely available on mobile platforms for estimating their 3D trajectories.
For this purpose, neural networks based on convolutional layers combined with a two-layer stacked bidirectional LSTM are explored from the following aspects.

# Pre-requisites
- torch: 1.5.0+cu101 (Just the CPU version is more than enough)
- matplotlib: 3.2.1
- numpy: 1.18.4
- PIL
- json

# Dataset 
The EuRoC MAV Dataset
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

# Inputs
IMU data :
Time squence of 200 timesteps (both past and future frames are used when computing the relative pose at each Δpose moment)

# Outputs
6-DOF Relative Pose (Generated absolute trajectory, using the starting absolute pose)


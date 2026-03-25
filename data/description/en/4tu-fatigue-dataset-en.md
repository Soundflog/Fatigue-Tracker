# Fatigue detection running dataset 

## Ref

[Dataset on 4tu](https://data.4tu.nl/datasets/d73599f0-cdf4-4dc8-8eec-5134bcebb53b)

## Description

Authors: L. Marotta
Roessingh Research & Development, Enschede
Biomedical Signals & Systems Group, EEMCS, University of Twente, Enschede

Corresponding author: L.Marotta
Contact Information:l.marotta@rrd.nl; l.marotta@utwente.nl

## ***General Introduction***
This dataset contains data collected during fatigue detection experiments in running using IMUs.Subjects underwent a fatiguing protocol consisting of three distinct consecutive runs on an athletic track:
1.	The first run consisted of a 4000 m run (10 laps) at a constant speed, determined as 100% of the average speed of the subject during the best performance in the previous year  on a 5 to 10 km race;
2.	The second run was performed according to a fatiguing protocol. 
The speed in this fatiguing protocol started at the same level of the first run and increased progres-sively of by 0.2km/h every 100 m. 
Perceived fatigue was assessed by means of a Borg Rating of Perceived Extertion (RPE) Scale (min-max score 6-20) [20], asked to the runner every 100 m. 
The fatiguing protocol was terminated once the RPE was higher than   16 (RPE between hard and very hard)  , or, if such requirement was not met, after 1200m;
3.	The third run consisted of a 1200m run (3 laps), in which speed was kept constant and equal to the first 4000 m run.

## ***Purpose of the test campaign***
The purpose of these experiments was to investigate the feasibility of fatigue detection in running using IMUs, and to assess the performance of different IMU locations.

## ***Test equipment***
Data were recorded with MVN Link system (Xsens, Enschede, The Netherlands) connected to 8 IMUs recording at 240 Hz.

## ***Description of the data in this data set***

Each of the 8 subjects measured in this study has 4 files:

- pXXX_XXX_0-2K: contains the Segment and Joint data exported from MVN for the first half of the first run
- pXXX_XXX_2-4K: contains the Segment and Joint data exported from MVN for the second half of the first run
- pXXX_XXX_postfatigue1200m: : contains the Segment and Joint data exported from MVN for the third run
- pXXX_strides: contains the segmented strides from each subject

Segment data are placed in a struct file named segment:
- segment.STE: sternum
- segment.PEL: pelvis
- segment.RUL: right thigh
- segment.LUL: left thigh
- segment.RLL: right tibia
- segment.LLL: left tibia
- segment.RFO: right foot
- segment.LFO: left foot

- segment.XXX.acc: acceleration data for a given segment (m/s^2)
- segment.XXX.angvel: angular velocity data for a given segment (rad/s)

Joint data are placed in a struct file named joint
- segment.L5S1: trunk
- segment.RHIP: right hip
- segment.LHIP: left hip
- segment.RKNE: right knee
- segment.LKNE: left knee
- segment.RANK: right ankle
- segment.LANK: left ankle

- segment.XXX.angle: angle data for a given joint (rad)

TableFeats: contains values used for the machine learning pipeline, after normalization over each single subject

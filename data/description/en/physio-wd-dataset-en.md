# Wearable Device Dataset from Induced Stress and Structured Exercise Sessions

## Reference
- [Dataset on PhysioNet](https://www.physionet.org/content/wearable-device-dataset/1.0.1/Wearable_Dataset/)

## FILES
- 'Stress_Level_v1.csv' and 'Stress_Level_v2.csv' contain self-reported stress levels during the stress protocol for each stage of the study,  respectively.
- 'subject-info.csv' contains demographic data such as age, weight, and height for all participants.
- 'data_constraints.txt' contains detailed information of issues such as incorrect wristband placements, incomplete protocols, and connection problems. 
- 'Wearable_Dataset.ipynb'  is a Jupyter Notebook that provides visualizations of all recorded signals and includes sample code for managing the data.
- The main folder 'Wearable_Dataset' contains three folders for each type of activity: 'STRESS', 'AEROBIC' and 'ANAEROBIC'.
- Each of these folders includes participant subfolders named as 'S01','S02','f01',etc.
- Each of the folders corresponding to each participant contains Empatica E4 files: 'ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'tags.csv' and 'TEMP.csv'.

## RELEVANT INFO
- Before downloading, we recommend reviewing the Jupyter Notebook ('Wearable_Dataset.ipynb') to determine whether the dataset fits your needs. This file is provided to read, open, visualize and start working with the data. To execute the notebook, ensure that basic Python libraries such as pandas,os, numpy, time, and matplotlib are installed.
- Participants from the first stage are labeled as 'Sxx' while those from the second stage are labeled as 'fxx'.
- Session dates and event marks are expressed in UTC and have been adjusted for de-identification. The time shifts have been applied consistently across all records to maintain signal alignment. 
- Empatica signal files: the first row is the initial time of the session expressed in UTC (Empatica provides time in Unix timestamp format, but files are already converted to UTC). The second row is the sample rate expressed in Hz.
- 'ACC.csv': data from x, y, and z axis are stored in first, second, and third column, respectively.
- 'IBI.csv': the first column is the time (respect to the initial time) of the detected inter-beat interval expressed in seconds (s). The second column is the duration in seconds (s) of the detected inter-beat interval (i.e., the distance in seconds from the previous beat).

## Methods

### Measurement Device
The Empatica E4 wristband is a wearable wireless device designed for continuous, real-time data acquisition. It includes a Photoplethysmography (PPG) Sensor (sampling frequency: 64 Hz) that measures blood volume pulse (BVP), from which heart rate (HR) may be derived; a 3-axis Accelerometer (sampling frequency: 32 Hz) for motion-based activity;an Infrared Thermopile (sampling frequency: 4 Hz) to read peripheral Skin Temperature and an Electrodermal Activity (EDA) Sensor (sampling frequency: 4 Hz) to measure sympathetic nervous system arousal. The E4 was worn on the subject's non-dominant hand to reduce motion artifacts when performing the tests.

Additionally, the Empatica button was used to mark events, which facilitates the identification and segmentation of each block of interest, enabling to easily identify and analyze different phases or sections of the protocol.

### Population
Enrollment in the study was managed through an online form. Exclusion criteria included individuals with chronic illnesses, a family history of sudden death during exercise, or those undergoing psychiatric treatment or taking medication that could impact physiological responses. Prior to participating in the tests, each participant signed an informed consent.

Participants were males and females aged between 18 and 30 years.

### Experimental design and data acquisition
The data collection process was conducted in two stages. Initially, a cohort of 18 volunteers (V1) followed the protocol. A few months later, a second group of 18 volunteers (V2) participated using an updated protocol that incorporated improvements based on initial experience. A detailed pipeline for each activity is shown in the provided Jupyter Notebook.

### STRESS INDUCEMENT PROTOCOL
The original protocol began with a 3-minute baseline recording to serve as a reference. The first stress test was an adaptation of the widely used Stroop Test [10,11], created using PsyToolkit [12,13]. Following this, participants had a 5-minute rest period.

Next experiment was a modified version of the Trier Mental Challenge Test [14], obtained through Millisecond Software, LLC. This test involved a series of mathematical tasks within a 5 seconds time limit, accompanied by an annoying sound stimulus.Participants were also instructed to vocalize their responses, increasing the cognitive load and performance demands.

Another 5-minute rest period preceded the final block, which comprised three tests, each with a time limit of 30 seconds. First, participants were asked to express their opinion about controversial topics and were then instructed to argue the opposite viewpoint, contrary to their true beliefs. Finally, participants counted backward from 1022 in decrements of 13, providing their answers aloud.

Before and after each task and rest period, participants verbally reported their self-perception stress level on a scale from 1 to 10.

- **AEROBIC PROTOCOL**
  
The aerobic exercise test is an adaptation of the Storer-Davis Maximal Bicycle Protocol, involving continuous stationary cycling for approximately 35 minutes [15].

Initially, we determined the maximum resistance for each participant by identifying the point at which they could no longer pedal with maximum effort. The protocol began with a 3-minute baseline recording during which the subject pedaled without resistance to warm up.

Following the warm up, the subject pedaled in sync with a metronome, where each beat corresponded to one foot down (or one knee up),completing one revolution every two beats. The exercise started with low resistance (20\% of maximum) and progressed through three 3-minute periods at increasing paces of 60, 70, and 75 rpm, with resistance gradually increasing to a medium level (30% of maximum).

The protocol then included four additional periods: the first two lasting 3 minutes each and the second two lasting 2 minutes each. These periods featured paces of 80, 85, 90, and 95 rpm, with a gradual increase in resistance (5% per stage).

The final phase implied a fixed medium-high resistance (50% of maximum) and consisted of three 2-minute periods at paces of 100, 105, and 110 rpm. This was followed by a 4-minute cooldown period with no resistance, and then 2 minutes of rest while remaining still.

- **ANAEROBIC PROTOCOL**
  
The anaerobic exercise test is an adaptation of the Wingate Anaerobic Test [16] . The protocol began with a 3-minute baseline recording during which the subject pedaled without resistance to warm up.

This was followed by three cycles, each consisting of 30 seconds of maximal effort, where the subject pedaled with utmost intensity against high resistance. Each cycle was succeeded by a 4-minute cooldown period with no resistance.

Finally, a 2-minute recording was made while the subject remained still.

### Protocols improvements:
The second version of the protocol incorporated several modifications based on previous experience. For stress induction, the Stroop Test was removed, and the second rest period was relocated between the opinion tasks and the subtraction test. Rest periods were extended and a relaxing video was shown . Additionally, the protocol was conducted remotely, which provided a more relaxed environment for rest periods.

In the updated exercise protocols, participants attended in groups to a spinning room.

The aerobic protocol was modified as follows: a baseline was introduced, followed by a 2:15-minute warm-up. This was succeeded by three 1:30-minute intervals at 70, 75, and 80 revolutions per minute (rpm), respectively. An 11:15-minute session at 85 rpm was conducted, leading into a final 4:30-minute period at 90/95 rpm (depending on the participant's condition). The session concluded with a 3-minute cooldown, followed by a rest period where participants sat on the bike without movement.

For the anaerobic protocol, a fourth maximum power sprint was added, with the sprints extended to 45 seconds each, followed by a corresponding cooldown period.

### Data conditioning
In accordance with HIPAA Safe Harbor De-Identification guidelines, each participant is assigned a unique ID. The session dates and event marks during the protocols (in tags.csv) have been modified by a random number of days, with the resulting shift exceeding one year. Time samples have been shifted consistently across all records to maintain signal alignment. Empatica provides time in Unix timestamp format, but files are already converted to UTC. Participants from the first stage are labeled as "Sxx" while those from the second stage are labeled as "fxx".


## Description

The dataset is organized into three main categories: STRESS, AEROBIC exercise, and ANAEROBIC exercise. Each category contains subfolders specific to individual subjects, where raw csv files downloaded from Empatica E4 Connect are stored. Only the "tags" files have been cleaned to improve protocol understanding. These tags mark the beginning and end of protocol segments, which facilitates signal segmentation.

Each subject folder contains the raw signal csv files provided by Empatica. These files follow this format: the first row is the initial time of the session expressed in UTC (Empatica provides time in Unix timestamp format, but files are already converted to UTC). The second row is the sample rate expressed in Hz.

TEMP.csv: Data from temperature sensor expressed degrees on the Celsius (°C) scale.
EDA.csv: Data from the electrodermal activity sensor expressed as microsiemens (μS).
BVP.csv: Data from photoplethysmograph.
ACC.csv: Data from 3-axis accelerometer sensor. The accelerometer is configured to measure acceleration in the range [-2g, 2g]. Therefore the unit in this file is 1/64g. Data from x, y, and z axis are respectively in first, second, and third column.
IBI.csv: Time between individuals heart beats extracted from the BVP signal. No sample rate is needed for this file. The first column is the time (respect to the initial time) of the detected inter-beat interval expressed in seconds (s). The second column is the duration in seconds (s) of the detected inter-beat interval (i.e., the distance in seconds from the previous beat).
HR.csv: Average heart rate extracted from the BVP signal.The first row is the initial time of the session expressed in UTC. The second row is the sample rate expressed in Hz.
tags.csv: Event mark times. Each row corresponds to a physical button press on the device; the same time as the status LED is first illuminated. The time is expressed in UTC and it is synchronized with initial time of the session indicated in the related data files from the corresponding session
Demographic data such as age, weight, and height are provided in the subject-info.csv file.

Additionally, a file containing all self-reported stress levels for each stage is provided (Stress_level_v1.csv and Stress_level_v2.csv).

Some participants experienced issues such as incorrect wristband placements, incomplete protocols, and connection problems. Details about these issues can be found in the data_constraints.txt file.



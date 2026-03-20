# WSD4FEDSRM (Wearable sensor data for fatigue estimation during shoulder rotation movements) 

The present study was financially supported by Science Foundation Ireland under Grant number 16/RC/3918 (CONFIRM) which is co-funded under the European Regional Development Fund.


#### Abstract

This data set was obtained using wearable sensors during a fatigue protocol involving dynamic shoulder internal rotation (IR) and external rotation (ER) movements. 
Thirty-four healthy volunteers performed shoulder IR and ER movements with 30-40%, 40-50%, and 50-60% of maximal voluntary isometric contraction (MVIC) force until 
reaching maximal effort on the Borg rating of perceived exertion (RPE) scale. The dataset comprises a collection of many data types, including demographic information, 
anthropometric measurements, MVIC force measurements, inertial measuring unit (IMU) data, surface electromyography (EMG) data, photoplethysmogram (PPG) data from wearable 
sensors, as well as measurements from the Borg RPE scale data and the Karolinska sleepiness scale (KSS) data. This detailed dataset offers valuable insights into fatigue assessment, 
enabling the development of fatigue detection algorithms and investigating human biomechanical characteristics during a fatigue protocol involving IR and ER shoulder motions.
 

### DESCRIPTION

The dataset comprises a collection of many data types during shoulder internal rotation (IR), and external rotation (ER) exercises from 34 participants, including demographic information, 
anthropometric measurements, maximum voluntary isometric contraction (MVIC) force measurements, inertial measuring unit (IMU) data, surface electromyography (EMG) data, 
photoplethysmogram (PPG) data, as well as measurements from the Borg rating of perceived exertion (RPE) scale and the Karolinska sleepiness scale (KSS).

 Please refer to the description.txt file for detailed information. 


### INCLUDED FILES

-README.txt: this .txt file.
-description.txt: properties of the dataset. 
-MVIC force data folder: 'MVIC force data'.
  The folder includes one file named 'MVIC_force_data.csv'
-Borg RPE scale data folder: 'Borg data'.
  The folder includes one file named 'borg_data.csv'
-KSS data folder: 'KSS data'.
  The folder includes one file named 'KSS_data.csv'
-Demographic and anthropometric data folder: 'Demographic and anthropometric data'.
  The folder consist of six .csv files named:
  *‘body_composition.csv’
  *‘breadth.csv’ 
  *‘circumference.csv’
  *‘demographic.csv’
  *‘depth.csv’
  *‘length .csv’
-EMG, IMU and PPG data folder: 'EMG, IMU and PPG data'
  The main folder consist of 10 subfolders named:
  *‘30–40% external rotation’ 
  *‘40–50% external rotation’ 
  *‘50–60% external rotation’ 
  *‘30–40% internal rotation’ 
  *‘40–50% internal rotation’ 
  *‘50–60% internal rotation’ 
  *‘MVIC force external rotation first’ 
  *‘MVIC force external rotation second’ 
  *‘MVIC force internal rotation first’ 
  *‘MVIC force external rotation first’  

#### Database Instructions:

### Inclusion criteria
•	Over the age of 18 years old
•	No musculoskeletal disorders
•	English speaking
•	Capacity to consent


### Participants Information

subject	        sex	age	height(cm)   dominant_hand
subject_1	female	31	165	     right
subject_2	female	23	163	     left
subject_3	male	26	183	     right
subject_4	male	25	188          right
subject_5	female	21	168	     right
subject_6	male	26	192	     right
subject_7	male	29	178	     right
subject_8	male	29	180	     right
subject_9	male	26	180	     left
subject_10	male	23	175	     right
subject_11	female	23	165	     right
subject_12	female	28	159	     right
subject_13	female	23	165	     right
subject_14	female	26	155	     right
subject_15	male	26	177	     right
subject_16	male	32	174	     right
subject_17	male	33	172	     right
subject_18	male	28	182	     right
subject_19	male 	29	161	     right
subject_20	male	27	180	     right
subject_21	female	21	167	     right
subject_22	male	26	195	     right
subject_23	female	23	174	     right
subject_24	male	28	180	     right
subject_25	male	21	175	     right
subject_26	male	26	180	     right
subject_27	male	31	195	     right
subject_28	female	20	170	     right
subject_29	male	35	170	     right
subject_30	male	29	172	     right
subject_31	male	20	182	     right
subject_32	female	32	164	     right
subject_33	male	21	177	     right
subject_34	male	29	186	     right


### Measurement tools 

-Surface EMG system (BTS FREEEMG 300, BTS Bioengineering, Italy) was used to record muscle electrical activity with a sampling frequency of 1000 Hz during MVIC force tests and shoulder IR and ER exercises.
 EMG sensors are located on the anterior deltoid, infraspinatus, posterior deltoid, upper trapezius, and latissimus dorsi muscles on the dominant side of the body.

-IMU system (MTw Awinda, Xsens Technologies B.V., Enschede, The Netherlands) was used to record accelerometer, gyroscope, and magnetometer data with a sampling frequency of 100 Hz during MVIC force tests and shoulder IR and ER exercises.
 IMU sensors are located on the hand, forearm, upper arm, shoulder, sternum, and pelvis on the dominant side of the body.

-PPG finger clip sensor (Biosignals Plux, Portugal) was used to record volumetric changes in blood circulation with a sampling frequency of 200 Hz during shoulder IR and ER exercises.
 PPG sensor is located on the non-dominant index finger.

-The Borg RPE scale was used to evaluate how easy or difficult a task is and how exhausted participants are when executing tasks at 10-second intervals during the shoulder IR and ER exercises.

-KSS was administered to participants at the very beginning and end of the measurements in order to evaluate their sleepiness.


### File naming conventions

-README.txt file


-description.txt file


-MVIC force data folder: 'MVIC force data'.
 The folder includes one file named 'MVIC_force_data.csv'

 The ‘MVIC_force_data.csv’ includes:

 ‘IR_MVIC_1_(N)’:       shoulder IR MVIC force first measurement.
 ‘IR_MVIC_2_ (N)’:      shoulder IR MVIC force second measurement.
 ‘IR_MVIC_mean_(N)’:    mean of first and second shoulder IR MVIC force.
 ‘ER_MVIC_1_(N)’:       shoulder ER MVIC force first measurement. 
 ‘ER_MVIC_2_ (N)’:      shoulder ER MVIC force second measurement.
 ‘ER_MVIC_mean_(N)’:    mean of first and second shoulder ER MVIC force.

-Borg RPE scale data folder: 'Borg data'.
 The folder includes one file named 'borg_data.csv'

 The ‘borg_data.csv’ includes (each subject's Borg RPE scale values during each exercise are given in this .csv file):

 ‘task_order’:             the sequence and load of the tasks 
                           To ensure a clear understanding for the reader, the task sequence does not correspond to the order in which the subject executed them 
                           (group one, two, and three); instead, it maintains an unvarying order.
     task1_35i:                 the shoulder IR exercise performed with a load in the 30-40% range of MVIC force.
     task2_45i:                 the shoulder IR exercise performed with a load in the 40-50% range of MVIC force.
     task3_55i:                 the shoulder IR exercise performed with a load in the 50-60% range of MVIC force.
     task4_35e:                 the shoulder ER exercise performed with a load in the 30-40% range of MVIC force.
     task5_45e:                 the shoulder ER exercise performed with a load in the 40-50% range of MVIC force.
     task6_55e:                 the shoulder ER exercise performed with a load in the 50-60% range of MVIC force.
 ‘before_task’:            the level of difficulty perceived by the participants before the exercises.
 ‘10_sec’:                 the participants' difficulty level at 10 seconds after the commencement of the exercise.
 ‘20_sec’:                 the participants' difficulty level at 20 seconds after the commencement of the exercise.
  .
  .
  .
  .
  .
 ‘250_sec’:                the participants' difficulty level at 250 seconds after the commencement of the exercise.
 ‘end_of_trial’:           the participants reached the level of 20 on the Borg RPE scale.
 ‘length_of_trial_(sec)’:  the duration of each exercise.


-KSS data folder: 'KSS data'.
 The folder includes one file named 'KSS_data.csv'
 
 The ‘KSS_data.csv’ includes (each subject's KSS values during each exercise are given in this .csv file):
 
 ‘KSS_before’:              KSS scores before the exercises.
 ‘KSS_after’:               KSS scores at the very end of the exercises. 


-demographic_anthropometric_data folder: The demographic and anthropometric measurements of the participants are provided in this folder. 
 Folder consist of six .csv files: ‘body_composition’, ‘breadth’,‘circumference’, ‘demographic’, ‘depth’ and ‘length’.


 ‘demographic.csv’ file includes:

  ‘date’:                                          date of the measurements (dd/mm/aaaa) 
  ‘time’:                                          the time of the measurements (24-hour clock).
  ‘group’:                                         the order of the exercises.
      Group	  Task order
      Group one:	  IR 30-40%, IR 40-50%, IR 50-60%, ER 30-40%, ER 40-50%, and ER 50-60%
      Group two:	  ER 50-60%, ER 40-50%, ER 30-40%, IR 50-60%, IR 40-50%, and IR 30-40%
      Group three:  IR 50-60%, ER 50-60%, IR 40-50%, ER 40-50%, IR 30-40%, and ER 30-40%
  ‘age’:                                           the age of the participants.
  ‘height(cm)’:                                    the height of the participants.
  ‘sex’:                                           the gender of the participants. 
  ‘dominant_hand’:                                 the dominant hand of the participants. 
  ‘what_kind_of_exercise_do_you_participate_in?’:  information about the type of exercises they engage in.
  ‘how_often_do_you_exercise_per_week?’:           information about the frequency with which they perform these exercises during the week.


 ‘body_composition.csv’ file includes:

  ‘mass(kg)’:      the body weight of the participants.  
  ‘body_fat%’:     the body fat ratio of the participants.
  ‘muscle%’:       the skeletal muscle mass of the participants.
  ‘visceral_fat’:  the visceral fat of the participants.
  ‘BMI(kg/m²)’:    the body mass index values of the participants.


 ‘length.csv’ file includes:

  ‘upperarm_l_(cm)’:    the upper arm length of participants.
  ‘forearm_l_(cm)’:     the forearm length of participants.
  ‘hand_l_(cm)’:        the hand length of participants.
  ‘palmar_l_(cm)’:      the palmar length of participants.


 ‘circumference.csv’ file includes:

  ‘upper_arm_mid_c_(cm)’:        the upper arm middle point circumference of participants. 
  ‘upper_arm_tense _c_(cm)’:     the upper arm tense circumference of participants.
  ‘upper_arm_distal_c_(cm)’:     the upper arm distal circumference of participants.
  ‘upper_arm_proximal_c_(cm)’:   the upper arm proximal circumference of participants.
  ‘forearm_mid_c_(cm)’:          the forearm middle point circumference of participants.    
  ‘forearm_distal_c_(cm)’:       the forearm distal circumference of participants.    
  ‘forearm_proximal_c_(cm)’:     the forearm proximal circumference
  ‘hand_c_(cm)’:                 the hand circumference of participants.


 ‘breadth.csv’ file includes:

  ‘upper_arm_mid_b_(cm)’:        the upper arm middle point breadth of participants.  
  ‘upper_arm_ditsal_b_(cm)':     the upper arm distal breadth of participants. 
  ‘upper_arm_proximal_b_(cm)’:   the upper arm proximal breadth of participants. 
  ‘forearm_mid_b_(cm)’:          the forearm middle point breadth of participants.
  ‘forearm_ditsal_b_(cm)’:       the forearm distal breadth of participants.
  ‘forearm_proximal_b_(cm)’:     the forearm proximal breadth of participants.
  ‘hand_b_(cm)’:                 the hand breadth of participants.


 ‘depth.csv’ file includes:     
 
  ‘upper_arm_mid_d_(cm)’:         the upper arm middle point depth of participants.  
  ‘upper_arm_ditsal_d_(cm)’:      the upper arm distal depth of participants.  
  ‘upper_arm_proximal_d_(cm)’:    the upper arm proximal depth of participants.
  ‘forearm_mid_d_(cm)’:           the forearm middle point depth of participants.
  ‘forearm_ditsal_d_(cm)’:        the forearm distal depth of participants.
  ‘forearm_proximal_d_(cm)’:      the forearm proximal depth of participants.
  ‘hand_d_(cm)’:                  the hand depth of participants.


-EMG, IMU and PPG data folder: the main folder consist of 10 folders named:

   *‘30–40% external rotation’ 
   *‘40–50% external rotation’ 
   *‘50–60% external rotation’ 
   *‘30–40% internal rotation’ 
   *‘40–50% internal rotation’ 
   *‘50–60% internal rotation’ 
   *‘MVIC force external rotation first’ 
   *‘MVIC force external rotation second’ 
   *‘MVIC force internal rotation first’ 
   *‘MVIC force external rotation first’

   *30-40% external rotation file folder: the shoulder ER exercise performed with a load in the 30-40% range of MVIC force.
    Folder consist of 34 subject (subject_x with x=partecipant number) folders.  

    **subject_x (with x from 1 to 34) folder.
      Each subject's directory contains EMG, IMU, and PPG data folder labelled as 'EMG data', 'IMU data', and 'PPG data'.

     ***EMG data folder consists of six excel .csv files:
      1) ‘anterior_deltoid.csv’:        EMG signals (mV) of the anterior deltoid muscle.
      2) ‘infraspinatus.csv’:           EMG signals (mV) of the infraspinatus muscle.
      3) ‘latissimus_dorsi.csv’:        EMG signals (mV) of the latissimus dorsi muscle.
      4) ‘pectoralis_major.csv’:        EMG signals (mV) of the pectoralis major muscle.         
      5) ‘posterior_deltoid.csv’:       EMG signals (mV) of the posterior deltoid muscle.
      6) ‘upper_trapezius.csv’:         EMG signals (mV) of the upper trapezius muscle.

     ***IMU data folder consists of six folders:
      1) Forearm consists of three .csv files:
         ‘acc_forearm.csv’:             acceleration data (m/s²) of the forearm. 
         ‘gyr_forearm.csv’:             gyrascope data (rad/s) of the forearm. 
         ‘mag_forearm.csv’:             magnotometer data (a.u.) of the forearm.
      2) Hand consists of three .csv files.
         ‘acc_hand.csv’:             acceleration data (m/s²) of the hand. 
         ‘gyr_hand.csv’:             gyrascope data (rad/s) of the hand. 
         ‘mag_hand.csv’:             magnotometer data (a.u.) of the hand.
      3) Pelvis consists of three .csv files.
         ‘acc_pelvis.csv’:             acceleration data (m/s²) of the pelvis. 
         ‘gyr_pelvis.csv’:             gyrascope data (rad/s) of the pelvis. 
         ‘mag_pelvis.csv’:             magnotometer data (a.u.) of the pelvis.
      4) Shoulder consists of three .csv files:
         ‘acc_shoulder.csv’:             acceleration data (m/s²) of the shoulder. 
         ‘gyr_shoulder.csv’:             gyrascope data (rad/s) of the shoulder. 
         ‘mag_shoulder.csv’:             magnotometer data (a.u.) of the shoulder.
      5) Sternum consists of three .csv files:
         ‘acc_sternum.csv’:             acceleration data (m/s²) of the sternum. 
         ‘gyr_sternum.csv’:             gyrascope data (rad/s) of the sternum. 
         ‘mag_sternum.csv’:             magnotometer data (a.u.) of the sternum.
      6) Upper arm consists of three .csv files:
         ‘acc_upper_arm.csv’:             acceleration data (m/s²) of the upper arm. 
         ‘gyr_upper_arm.csv’:             gyrascope data (rad/s) of the upper arm. 
         ‘mag_upper_arm.csv’:             magnotometer data (a.u.) of the upper arm.
 
     ***PPG data folder consists of one .csv files:
      1) ‘ppg.csv’:                      : ppg data (V) during the exercise. 


   *30-40% internal rotation file folder:               the shoulder IR exercise performed with a load in the 30-40% range of MVIC force.


   *40-50% external rotation file folder:               the shoulder ER exercise performed with a load in the 40-50% range of MVIC force.
  

   *40-50% internal rotation file folder:               the shoulder IR exercise performed with a load in the 40-50% range of MVIC force.


   *50-60% external rotation file folder:               the shoulder ER exercise performed with a load in the 50-60% range of MVIC force. 


   *50-60% internal rotation file folder:               the shoulder IR exercise performed with a load in the 50-60% range of MVIC force. 


   *MVIC force external rotation first file folder:     the shoulder ER MVIC force first measurement (PPG data are not included in the exercises related to the MVIC force).


   *MVIC force external rotation second file folder:    the shoulder ER MVIC force second measurement (PPG data are not included in the exercises related to the MVIC force).


   *MVIC force internal rotation first file folder:     the shoulder IR MVIC force first measurement (PPG data are not included in the exercises related to the MVIC force).


   *MVIC force internal rotation second file folder:    the shoulder IR MVIC force second measurement (PPG data are not included in the exercises related to the MVIC force).

# WSD4FEDSRM (Wearable sensor data for fatigue estimation during shoulder rotation movements) 

The present study was financially supported by Science Foundation Ireland under Grant number 16/RC/3918 (CONFIRM) which is co-funded under the European Regional Development Fund.

## Reference

[Dataset from zenodo](https://zenodo.org/records/8415066)

## Abstract

This data set was obtained using wearable sensors during a fatigue protocol involving dynamic shoulder internal rotation (IR) and external rotation (ER) movements. 
Thirty-four healthy volunteers performed shoulder IR and ER movements with 30-40%, 40-50%, and 50-60% of maximal voluntary isometric contraction (MVIC) force until 
reaching maximal effort on the Borg rating of perceived exertion (RPE) scale. The dataset comprises a collection of many data types, including demographic information, 
anthropometric measurements, MVIC force measurements, inertial measuring unit (IMU) data, surface electromyography (EMG) data, photoplethysmogram (PPG) data from wearable 
sensors, as well as measurements from the Borg RPE scale data and the Karolinska sleepiness scale (KSS) data. This detailed dataset offers valuable insights into fatigue assessment, 
enabling the development of fatigue detection algorithms and investigating human biomechanical characteristics during a fatigue protocol involving IR and ER shoulder motions.
 

## DESCRIPTION

The dataset comprises a collection of many data types during shoulder internal rotation (IR), and external rotation (ER) exercises from 34 participants, including demographic information, 
anthropometric measurements, maximum voluntary isometric contraction (MVIC) force measurements, inertial measuring unit (IMU) data, surface electromyography (EMG) data, 
photoplethysmogram (PPG) data, as well as measurements from the Borg rating of perceived exertion (RPE) scale and the Karolinska sleepiness scale (KSS).

 Please refer to the description.txt file for detailed information. 


## INCLUDED FILES

- README.txt: this .txt file.
- description.txt: properties of the dataset.
- **MVIC force data** folder: `MVIC force data`
   - Contains `MVIC_force_data.csv`
- **Borg RPE scale data** folder: `Borg data`
   - Contains `borg_data.csv`
- **KSS data** folder: `KSS data`
   - Contains `KSS_data.csv`
- **Demographic and anthropometric data** folder: `Demographic and anthropometric data`
   - `body_composition.csv`
   - `breadth.csv`
   - `circumference.csv`
   - `demographic.csv`
   - `depth.csv`
   - `length.csv`
- **EMG, IMU and PPG data** folder: `EMG, IMU and PPG data`
   - `30–40% external rotation`
   - `40–50% external rotation`
   - `50–60% external rotation`
   - `30–40% internal rotation`
   - `40–50% internal rotation`
   - `50–60% internal rotation`
   - `MVIC force external rotation first`
   - `MVIC force external rotation second`
   - `MVIC force internal rotation first`
   - `MVIC force internal rotation second`

## Database Instructions:

### Inclusion criteria

-	Over the age of 18 years old
-	No musculoskeletal disorders
-	English speaking
-	Capacity to consent


### Participants Information

table of demographic and anthropometric data of the participants is given in the 'demographic.csv' file in the 'Demographic and anthropometric data' folder.
A table of demographic data for all 34 participants is provided below:

| Subject | Sex | Age | Height (cm) | Dominant Hand |
|---------|-----|-----|-------------|---------------|
| subject_1 | Female | 31 | 165 | Right |
| subject_2 | Female | 23 | 163 | Left |
| subject_3 | Male | 26 | 183 | Right |
| subject_4 | Male | 25 | 188 | Right |
| subject_5 | Female | 21 | 168 | Right |
| subject_6 | Male | 26 | 192 | Right |
| subject_7 | Male | 29 | 178 | Right |
| subject_8 | Male | 29 | 180 | Right |
| subject_9 | Male | 26 | 180 | Left |
| subject_10 | Male | 23 | 175 | Right |
| subject_11 | Female | 23 | 165 | Right |
| subject_12 | Female | 28 | 159 | Right |
| subject_13 | Female | 23 | 165 | Right |
| subject_14 | Female | 26 | 155 | Right |
| subject_15 | Male | 26 | 177 | Right |
| subject_16 | Male | 32 | 174 | Right |
| subject_17 | Male | 33 | 172 | Right |
| subject_18 | Male | 28 | 182 | Right |
| subject_19 | Male | 29 | 161 | Right |
| subject_20 | Male | 27 | 180 | Right |
| subject_21 | Female | 21 | 167 | Right |
| subject_22 | Male | 26 | 195 | Right |
| subject_23 | Female | 23 | 174 | Right |
| subject_24 | Male | 28 | 180 | Right |
| subject_25 | Male | 21 | 175 | Right |
| subject_26 | Male | 26 | 180 | Right |
| subject_27 | Male | 31 | 195 | Right |
| subject_28 | Female | 20 | 170 | Right |
| subject_29 | Male | 35 | 170 | Right |
| subject_30 | Male | 29 | 172 | Right |
| subject_31 | Male | 20 | 182 | Right |
| subject_32 | Female | 32 | 164 | Right |
| subject_33 | Male | 21 | 177 | Right |
| subject_34 | Male | 29 | 186 | Right |



## Measurement tools 

- Surface EMG system (BTS FREEEMG 300, BTS Bioengineering, Italy) was used to record muscle electrical activity with a sampling frequency of 1000 Hz during MVIC force tests and shoulder IR and ER exercises.
  - EMG sensors are located on the anterior deltoid, infraspinatus, posterior deltoid, upper trapezius, and latissimus dorsi muscles on the dominant side of the body.

- IMU system (MTw Awinda, Xsens Technologies B.V., Enschede, The Netherlands) was used to record accelerometer, gyroscope, and magnetometer data with a sampling frequency of 100 Hz during MVIC force tests and shoulder IR and ER exercises.
  - IMU sensors are located on the hand, forearm, upper arm, shoulder, sternum, and pelvis on the dominant side of the body.

- PPG finger clip sensor (Biosignals Plux, Portugal) was used to record volumetric changes in blood circulation with a sampling frequency of 200 Hz during shoulder IR and ER exercises.
 - PPG sensor is located on the non-dominant index finger.

- The Borg RPE scale was used to evaluate how easy or difficult a task is and how exhausted participants are when executing tasks at 10-second intervals during the shoulder IR and ER exercises.

- KSS was administered to participants at the very beginning and end of the measurements in order to evaluate their sleepiness.


## File naming conventions

- README.txt file

- description.txt file

### MVIC force data folder: 'MVIC force data'.

- The folder includes one file named 'MVIC_force_data.csv'

- The ‘MVIC_force_data.csv’ includes:
   - `IR_MVIC_1_(N)`:       shoulder IR MVIC force first measurement.
   - `IR_MVIC_2_ (N)`:      shoulder IR MVIC force second measurement.
   - `IR_MVIC_mean_(N)`:    mean of first and second shoulder IR MVIC force.
   - `ER_MVIC_1_(N)`:       shoulder ER MVIC force first measurement. 
   - `ER_MVIC_2_ (N)`:      shoulder ER MVIC force second measurement.
   - `ER_MVIC_mean_(N)`:    mean of first and second shoulder ER MVIC force.

### Borg RPE scale data folder: `Borg data`.

The folder includes one file named `borg_data.csv`

The `borg_data.csv` includes (each subject's Borg RPE scale values during each exercise are given in this .csv file):

| Column | Description |
|--------|-------------|
| `task_order` | The sequence and load of the tasks. Note: the task sequence does not correspond to the actual execution order (group one, two, three) but maintains a consistent order. |
| `task1_35i` | Shoulder IR exercise at 30-40% MVIC force |
| `task2_45i` | Shoulder IR exercise at 40-50% MVIC force |
| `task3_55i` | Shoulder IR exercise at 50-60% MVIC force |
| `task4_35e` | Shoulder ER exercise at 30-40% MVIC force |
| `task5_45e` | Shoulder ER exercise at 40-50% MVIC force |
| `task6_55e` | Shoulder ER exercise at 50-60% MVIC force |
| `before_task` | Perceived difficulty level before exercises |
| `10_sec` to `250_sec` | Difficulty level at 10-second intervals during exercise |
| `end_of_trial` | Indicates when participants reached Borg RPE scale level 20 |
| `length_of_trial_(sec)` | Duration of each exercise in seconds |


### KSS data

The folder includes one file named 'KSS_data.csv'

The ‘KSS_data.csv’ includes (each subject's KSS values during each exercise are given in this .csv file):

| Column | Description |
|--------|-------------|
| `KSS_before` | KSS scores before the exercises |
| `KSS_after` | KSS scores at the very end of the exercises |


### Demographic and anthropometric data folder: 'demographic_anthropometric_data'.

The folder includes six .csv files: ‘body_composition’, ‘breadth’,‘circumference’, ‘demographic’, ‘depth’ and ‘length’.


| Column | Description |
|--------|-------------|
| `date` | Date of the measurements (dd/mm/yyyy) |
| `time` | The time of the measurements (24-hour clock) |
| `group` | The order of the exercises |
| `age` | The age of the participants |
| `height(cm)` | The height of the participants |
| `sex` | The gender of the participants |
| `dominant_hand` | The dominant hand of the participants |
| `what_kind_of_exercise_do_you_participate_in?` | Information about the type of exercises they engage in |
| `how_often_do_you_exercise_per_week?` | Information about the frequency with which they perform these exercises during the week |

**Group Exercise Orders:**

| Group | Task Order |
|-------|-----------|
| Group one | IR 30-40%, IR 40-50%, IR 50-60%, ER 30-40%, ER 40-50%, ER 50-60% |
| Group two | ER 50-60%, ER 40-50%, ER 30-40%, IR 50-60%, IR 40-50%, IR 30-40% |
| Group three | IR 50-60%, ER 50-60%, IR 40-50%, ER 40-50%, IR 30-40%, ER 30-40% |

### body_composition.csv

| Column | Description |
|--------|-------------|
| `mass(kg)` | The body weight of the participants |
| `body_fat%` | The body fat ratio of the participants |
| `muscle%` | The skeletal muscle mass of the participants |
| `visceral_fat` | The visceral fat of the participants |
| `BMI(kg/m²)` | The body mass index values of the participants |

### length.csv

| Column | Description |
|--------|-------------|
| `upperarm_l_(cm)` | The upper arm length of participants |
| `forearm_l_(cm)` | The forearm length of participants |
| `hand_l_(cm)` | The hand length of participants |
| `palmar_l_(cm)` | The palmar length of participants |

### circumference.csv

| Column | Description |
|--------|-------------|
| `upper_arm_mid_c_(cm)` | The upper arm middle point circumference of participants |
| `upper_arm_tense_c_(cm)` | The upper arm tense circumference of participants |
| `upper_arm_distal_c_(cm)` | The upper arm distal circumference of participants |
| `upper_arm_proximal_c_(cm)` | The upper arm proximal circumference of participants |
| `forearm_mid_c_(cm)` | The forearm middle point circumference of participants |
| `forearm_distal_c_(cm)` | The forearm distal circumference of participants |
| `forearm_proximal_c_(cm)` | The forearm proximal circumference of participants |
| `hand_c_(cm)` | The hand circumference of participants |

### breadth.csv

| Column | Description |
|--------|-------------|
| `upper_arm_mid_b_(cm)` | The upper arm middle point breadth of participants |
| `upper_arm_distal_b_(cm)` | The upper arm distal breadth of participants |
| `upper_arm_proximal_b_(cm)` | The upper arm proximal breadth of participants |
| `forearm_mid_b_(cm)` | The forearm middle point breadth of participants |
| `forearm_distal_b_(cm)` | The forearm distal breadth of participants |
| `forearm_proximal_b_(cm)` | The forearm proximal breadth of participants |
| `hand_b_(cm)` | The hand breadth of participants |

### depth.csv

| Column | Description |
|--------|-------------|
| `upper_arm_mid_d_(cm)` | The upper arm middle point depth of participants |
| `upper_arm_distal_d_(cm)` | The upper arm distal depth of participants |
| `upper_arm_proximal_d_(cm)` | The upper arm proximal depth of participants |
| `forearm_mid_d_(cm)` | The forearm middle point depth of participants |
| `forearm_distal_d_(cm)` | The forearm distal depth of participants |
| `forearm_proximal_d_(cm)` | The forearm proximal depth of participants |
| `hand_d_(cm)` | The hand depth of participants |


### EMG, IMU and PPG data folder: 

| Folder Name | Description |
|-------------|-------------|
| `30–40% external rotation` | Shoulder ER exercise at 30-40% MVIC force. Contains 34 subject folders with EMG, IMU, and PPG data. |
| `40–50% external rotation` | Shoulder ER exercise at 40-50% MVIC force. Contains 34 subject folders with EMG, IMU, and PPG data. |
| `50–60% external rotation` | Shoulder ER exercise at 50-60% MVIC force. Contains 34 subject folders with EMG, IMU, and PPG data. |
| `30–40% internal rotation` | Shoulder IR exercise at 30-40% MVIC force. Contains 34 subject folders with EMG, IMU, and PPG data. |
| `40–50% internal rotation` | Shoulder IR exercise at 40-50% MVIC force. Contains 34 subject folders with EMG, IMU, and PPG data. |
| `50–60% internal rotation` | Shoulder IR exercise at 50-60% MVIC force. Contains 34 subject folders with EMG, IMU, and PPG data. |
| `MVIC force external rotation first` | Shoulder ER MVIC force first measurement. PPG data not included. |
| `MVIC force external rotation second` | Shoulder ER MVIC force second measurement. PPG data not included. |
| `MVIC force internal rotation first` | Shoulder IR MVIC force first measurement. PPG data not included. |
| `MVIC force internal rotation second` | Shoulder IR MVIC force second measurement. PPG data not included. |

**Subject Folder Structure (subject_x, x = 1-34):**

| Folder | Files | Description |
|--------|-------|-------------|
| `EMG data` | `anterior_deltoid.csv`, `infraspinatus.csv`, `latissimus_dorsi.csv`, `pectoralis_major.csv`, `posterior_deltoid.csv`, `upper_trapezius.csv` | EMG signals (mV) for each muscle |
| `IMU data` - `Forearm` | `acc_forearm.csv`, `gyr_forearm.csv`, `mag_forearm.csv` | Acceleration (m/s²), gyroscope (rad/s), magnetometer (a.u.) |
| `IMU data` - `Hand` | `acc_hand.csv`, `gyr_hand.csv`, `mag_hand.csv` | Acceleration (m/s²), gyroscope (rad/s), magnetometer (a.u.) |
| `IMU data` - `Pelvis` | `acc_pelvis.csv`, `gyr_pelvis.csv`, `mag_pelvis.csv` | Acceleration (m/s²), gyroscope (rad/s), magnetometer (a.u.) |
| `IMU data` - `Shoulder` | `acc_shoulder.csv`, `gyr_shoulder.csv`, `mag_shoulder.csv` | Acceleration (m/s²), gyroscope (rad/s), magnetometer (a.u.) |
| `IMU data` - `Sternum` | `acc_sternum.csv`, `gyr_sternum.csv`, `mag_sternum.csv` | Acceleration (m/s²), gyroscope (rad/s), magnetometer (a.u.) |
| `IMU data` - `Upper arm` | `acc_upper_arm.csv`, `gyr_upper_arm.csv`, `mag_upper_arm.csv` | Acceleration (m/s²), gyroscope (rad/s), magnetometer (a.u.) |
| `PPG data` | `ppg.csv` | PPG data (V) during exercise |

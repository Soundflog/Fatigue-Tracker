*** Fatigue detection running dataset ***

Authors: L. Marotta
Roessingh Research & Development, Enschede
Biomedical Signals & Systems Group, EEMCS, University of Twente, Enschede

Corresponding author: L.Marotta
Contact Information:l.marotta@rrd.nl; l.marotta@utwente.nl

***General Introduction***
This dataset contains data collected during fatigue detection experiments in running using IMUs.Subjects underwent a fatiguing protocol consisting of three distinct consecutive runs on an athletic track:
1.	The first run consisted of a 4000 m run (10 laps) at a constant speed, determined as 100% of the average speed of the subject during the best performance in the previous year  on a 5 to 10 km race;
2.	The second run was performed according to a fatiguing protocol. The speed in this fatiguing protocol started at the same level of the first run and increased progres-sively of by 0.2km/h every 100 m. Perceived fatigue was assessed by means of a Borg Rating of Perceived Extertion (RPE) Scale (min-max score 6-20) [20], asked to the runner every 100 m. The fatiguing protocol was terminated once the RPE was higher than   16 (RPE between hard and very hard)  , or, if such requirement was not met, after 1200m;
3.	The third run consisted of a 1200m run (3 laps), in which speed was kept constant and equal to the first 4000 m run.

***Purpose of the test campaign***
The purpose of these experiments was to investigate the feasibility of fatigue detection in running using IMUs, and to assess the performance of different IMU locations.

***Test equipment***
Data were recorded with MVN Link system (Xsens, Enschede, The Netherlands) connected to 8 IMUs recording at 240 Hz.

***Description of the data in this data set***

Each of the 8 subjects measured in this study has 4 files:

-pXXX_XXX_0-2K: contains the Segment and Joint data exported from MVN for the first half of the first run
-pXXX_XXX_2-4K: contains the Segment and Joint data exported from MVN for the second half of the first run
-pXXX_XXX_postfatigue1200m: : contains the Segment and Joint data exported from MVN for the third run
-pXXX_strides: contains the segmented strides from each subject

Segment data are placed in a struct file named segment:
-segment.STE: sternum
-segment.PEL: pelvis
-segment.RUL: right thigh
-segment.LUL: left thigh
-segment.RLL: right tibia
-segment.LLL: left tibia
-segment.RFO: right foot
-segment.LFO: left foot

-segment.XXX.acc: acceleration data for a given segment (m/s^2)
-segment.XXX.angvel: angular velocity data for a given segment (rad/s)

Joint data are placed in a struct file named joint
-segment.L5S1: trunk
-segment.RHIP: right hip
-segment.LHIP: left hip
-segment.RKNE: right knee
-segment.LKNE: left knee
-segment.RANK: right ankle
-segment.LANK: left ankle

-segment.XXX.angle: angle data for a given joint (rad)

TableFeats: contains values used for the machine learning pipeline, after normalization over each single subject

---

*** Fatigue detection running dataset ***

Authors: L. Marotta
Roessingh Research & Development, Enschede
Biomedical Signals & Systems Group, EEMCS, University of Twente, Enschede

Corresponding author: L.Marotta
Contact Information:l.marotta@rrd.nl; l.marotta@utwente.nl

Этот набор данных содержит информацию, собранную в ходе экспериментов по определению усталости при беге с использованием инерциальных измерительных устройств (IMU). Испытуемые прошли протокол, направленный на вызывание усталости, состоящий из трех последовательных забегов на легкоатлетической дорожке:
1. Первый забег состоял из бега на 4000 м (10 кругов) с постоянной скоростью, равной 100 % от средней скорости испытуемого во время лучшего результата в предыдущем году  на дистанции от 5 до 10 км;
2. Второй забег выполнялся в соответствии с протоколом усталости. 
Скорость в этом протоколе усталости начиналась на том же уровне, что и в первом забеге, и постепенно увеличивалась на 0,2 км/ч каждые 100 м. 
Воспринимаемая усталость оценивалась с помощью шкалы воспринимаемой нагрузки по Боргу (RPE) (минимальный-максимальный балл 6-20) [20], которую бегуну задавали каждые 100 м. 
Протокол утомления прекращался, когда RPE превышал   16 (RPE между тяжелым и очень тяжелым)  , или, если такое требование не было выполнено, после 1200 м;
3.    Третий забег состоял из бега на 1200 м (3 круга), в котором скорость оставалась постоянной и равной скорости первого забега на 4000 м.


***Цель тестовой кампании***
Целью этих экспериментов было изучение возможности обнаружения усталости при беге с помощью IMU и оценка эффективности различных расположений IMU.

***Тестовое оборудование***
Данные регистрировались с помощью системы MVN Link (Xsens, Энсхеде, Нидерланды), подключенной к 8 IMU, регистрирующим данные с частотой 240 Гц.

***Описание данных в этом наборе данных***

Каждый из 8 испытуемых, участвовавших в этом исследовании, имеет 4 файла:

- pXXX_XXX_0-2K: содержит данные о сегментах и суставах, экспортированные из MVN для первой половины первого забега
- pXXX_XXX_2-4K: содержит данные о сегментах и суставах, экспортированные из MVN для второй половины первого забега
- pXXX_XXX_postfatigue1200m: содержит данные о сегментах и суставах, экспортированные из MVN для третьего забега
- pXXX_strides: содержит сегментированные шаги каждого испытуемого

Данные о сегментах помещены в файл структуры с именем segment:
- segment.STE: грудина
- segment.PEL: таз
- segment.RUL: правое бедро
- segment.LUL: левое бедро
- segment.RLL: правая большеберцовая кость
- segment.LLL: левая большеберцовая кость
- segment.RFO: правая стопа
- segment.LFO: левая стопа

- segment.XXX.acc: данные об ускорении для данного сегмента (м/с^2)
- segment.XXX.angvel: данные об угловой скорости для данного сегмента (рад/с)

Данные о суставах помещаются в файл структуры с именем joint
- segment.L5S1: туловище
- segment.RHIP: правое бедро
- segment.LHIP: левое бедро
- segment.RKNE: правое колено
- segment.LKNE: левое колено
- segment.RANK: правый голеностоп
- segment.LANK: левый голеностоп
  
- segment.XXX.angle: данные об угле для данного сустава (рад)

TableFeats: содержит значения, используемые для конвейера машинного обучения, после нормализации по каждому отдельному субъекту
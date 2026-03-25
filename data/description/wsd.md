# WSD4FEDSRM (Wearable sensor data for fatigue estimation during shoulder rotation movements) 

## Reference

[Dataset from zenodo](https://zenodo.org/records/8415066)

## Abstract

Данные были получены с помощью носимых датчиков в ходе протокола по изучению усталости, включавшего динамические движения плеча на внутреннее (IR) и внешнее (ER) вращение.

Тридцать четыре здоровых добровольца выполняли движения IR и ER плеча с усилием, составляющим 30–40 %, 40–50 % и 50–60 % от силы максимального добровольного изометрического сокращения (MVIC), до 
достижения максимального усилия по шкале воспринимаемой нагрузки (RPE) по Боргу. 

Набор данных включает в себя множество типов данных, в том числе демографическую информацию, 
антропометрические измерения, измерения силы MVIC, данные инерциальной измерительной системы (IMU), данные поверхностной электромиографии (EMG), данные фотоплетизмограммы (PPG) с носимых 
датчиков, а также данные по шкале RPE по Боргу и данные по шкале сонливости Каролинского института (KSS).

Этот подробный набор данных дает ценную информацию для оценки усталости, 
позволяя разрабатывать алгоритмы обнаружения усталости и исследовать биомеханические характеристики человека во время протокола оценки усталости, включающего движения плеча в диапазонах IR и ER.


## DESCRIPTION

Набор данных включает в себя совокупность данных различных типов, полученных во время выполнения упражнений на внутреннее (IR) и внешнее (ER) вращение плеча 34 участниками, в том числе демографическую информацию, 
антропометрические измерения, результаты измерения силы максимального добровольного изометрического сокращения (MVIC), данные инерциальной измерительной системы (IMU), данные поверхностной электромиографии (EMG), 
данные фотоплетизмограммы (PPG), а также результаты измерений по шкале воспринимаемой нагрузки Борга (RPE) и шкале сонливости Каролинского института (KSS).

 Подробную информацию см. в файле description.txt.

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

## Database Instructions

### Inclusion criteria

-    Возраст старше 18 лет
-    Отсутствие заболеваний опорно-двигательного аппарата
-    Владение английским языком
-    Способность давать согласие

### Participants Information

Таблица демографических и антропометрических данных участников приведена в файле «demographic.csv», расположенном в папке «Демографические и антропометрические данные».
Ниже приведена таблица демографических данных всех 34 участников:

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

- Для регистрации электрической активности мышц с частотой дискретизации 1000 Гц во время силовых тестов MVIC и упражнений на внутреннее и внешнее вращение плеча использовалась система поверхностной ЭМГ (BTS FREEEMG 300, BTS Bioengineering, Италия).
  - Датчики ЭМГ располагались на передней, задней и подлопаточной мышцах дельтовидного мышечного комплекса, верхней части трапециевидной мышцы и широчайшей мышце спины на доминирующей стороне тела.

- Система IMU (MTw Awinda, Xsens Technologies B.V., Энсхеде, Нидерланды) использовалась для регистрации данных акселерометра, гироскопа и магнитометра с частотой дискретизации 100 Гц во время силовых тестов MVIC и упражнений на внутреннее и внешнее вращение плеча.
  - Датчики IMU расположены на кисти, предплечье, плече, грудине и тазе на доминирующей стороне тела.

- Датчик PPG в виде зажима на палец (Biosignals Plux, Португалия) использовался для регистрации объемных изменений в кровообращении с частотой дискретизации 200 Гц во время упражнений на внутреннее и внешнее вращение плеча.
  - Датчик PPG расположен на указательном пальце не доминирующей руки.

- Шкала RPE по Боргу использовалась для оценки того, насколько легко или сложно выполняется задача, а также степени утомления участников при выполнении задач с интервалом в 10 секунд во время упражнений на внутреннее и внешнее вращение плеча.

- Участникам предлагалось заполнить шкалу KSS в самом начале и в конце измерений для оценки их сонливости.

## File naming conventions

- README.txt file

- description.txt file

### MVIC force data folder: 'MVIC force data'.

- The folder includes one file named 'MVIC_force_data.csv'

- The ‘MVIC_force_data.csv’ includes:
   - `IR_MVIC_1_(N)`:      первое измерение силы IR MVIC плеча.
   - `IR_MVIC_2_ (N)`:      второе измерение силы IR MVIC плеча.
   - `IR_MVIC_mean_(N)`:    среднее значение первого и второго измерений силы IR MVIC плеча.
   - `ER_MVIC_1_(N)`:       первое измерение силы ER MVIC плеча. 
   - `ER_MVIC_2_ (N)`:      второе измерение силы ER MVIC плеча.
   - `ER_MVIC_mean_(N)`:    среднее значение первого и второго измерений силы ER MVIC плеча.

### Borg RPE scale data folder: `Borg data`.

The folder includes one file named `borg_data.csv`

The `borg_data.csv` includes (each subject's Borg RPE scale values during each exercise are given in this .csv file):

| Column | Description |
|--------|-------------|
| `task_order` | Порядок и нагрузка задач. Примечание: последовательность задач не соответствует фактическому порядку выполнения (группа один, два, три), но сохраняет последовательный порядок. |
| `task1_35i` | Упражнение на внутреннее вращение плеча при 30-40% силы MVIC |
| `task2_45i` | Упражнение на внутреннее вращение плеча при 40-50% силы MVIC |
| `task3_55i` | Упражнение на внутреннее вращение плеча при 50-60% силы MVIC |
| `task4_35e` | Упражнение на внешнее вращение плеча при 30-40% силы MVIC |
| `task5_45e` | Упражнение на внешнее вращение плеча при 40-50% силы MVIC |
| `task6_55e` | Упражнение на внешнее вращение плеча при 50-60% силы MVIC |
| `before_task` | Уровень воспринимаемой сложности перед упражнениями |
| `10_sec` to `250_sec` | Уровень сложности через каждые 10 секунд во время упражнения |
| `end_of_trial` | Указывает, когда участники достигли уровня 20 по шкале Borg RPE |
| `length_of_trial_(sec)` | Продолжительность каждого упражнения в секундах |


### KSS data

The folder includes one file named 'KSS_data.csv'

The ‘KSS_data.csv’ includes (each subject's KSS values during each exercise are given in this .csv file):

| Column | Description |
|--------|-------------|
| `KSS_before` | Значения KSS перед упражнениями |
| `KSS_after` | Значения KSS в самом конце упражнений |


### Demographic and anthropometric data folder: 'demographic_anthropometric_data'.

The folder includes six .csv files: ‘body_composition’, ‘breadth’,‘circumference’, ‘demographic’, ‘depth’ and ‘length’.


| Column | Description |
|--------|-------------|
| `date` | Дата измерений (дд/мм/гггг) |
| `time` | Время измерений (24-часовой формат) |
| `group` | Порядок выполнения упражнений |
| `age` | Возраст участников |
| `height(cm)` | Рост участников |
| `sex` | Пол участников |
| `dominant_hand` | Доминирующая рука участников |
| `what_kind_of_exercise_do_you_participate_in?` | Информация о типе упражнений, в которых они участвуют |
| `how_often_do_you_exercise_per_week?` | Информация о частоте выполнения этих упражнений в течение недели |

**Порядок выполнения упражнений по группам:**

| Группа | Порядок выполнения упражнений |
|-------|-----------------------------|
| Группа один | IR 30-40%, IR 40-50%, IR 50-60%, ER 30-40%, ER 40-50%, ER 50-60% |
| Группа два | ER 50-60%, ER 40-50%, ER 30-40%, IR 50-60%, IR 40-50%, IR 30-40% |
| Группа три | IR 50-60%, ER 50-60%, IR 40-50%, ER 40-50%, IR 30-40%, ER 30-40% |

### body_composition.csv

| Column | Description |
|--------|-------------|
| `mass(kg)` | Масса тела участников |
| `body_fat%` | Процент жира в организме участников |
| `muscle%` | Процент скелетной мышечной массы участников |
| `visceral_fat` | Висцеральный жир участников |
| `BMI(kg/m²)` | Индекс массы тела участников |

### length.csv

| Column | Description |
|--------|-------------|
| `upperarm_l_(cm)` | Длина верхней части руки участников |
| `forearm_l_(cm)` | Длина предплечья участников |
| `hand_l_(cm)` | Длина кисти участников |
| `palmar_l_(cm)` | Длина ладони участников |

### circumference.csv

| Column | Description |
|--------|-------------|
| `upper_arm_mid_c_(cm)` | Окружность средней части верхней руки участников |
| `upper_arm_tense_c_(cm)` | Окружность напряженной верхней руки участников |
| `upper_arm_distal_c_(cm)` | Окружность дистальной части верхней руки участников |
| `upper_arm_proximal_c_(cm)` | Окружность проксимальной части верхней руки участников |
| `forearm_mid_c_(cm)` | Окружность средней части предплечья участников |
| `forearm_distal_c_(cm)` | Окружность дистальной части предплечья участников |
| `forearm_proximal_c_(cm)` | Окружность проксимальной части предплечья участников |
| `hand_c_(cm)` | Окружность кисти участников |

### breadth.csv

| Column | Description |
|--------|-------------|
| `upper_arm_mid_b_(cm)` | Ширина средней части верхней руки участников |
| `upper_arm_distal_b_(cm)` | Ширина дистальной части верхней руки участников |
| `upper_arm_proximal_b_(cm)` | Ширина проксимальной части верхней руки участников |
| `forearm_mid_b_(cm)` | Ширина средней части предплечья участников |
| `forearm_distal_b_(cm)` | Ширина дистальной части предплечья участников |
| `forearm_proximal_b_(cm)` | Ширина проксимальной части предплечья участников |
| `hand_b_(cm)` | Ширина кисти участников |

### depth.csv

| Column | Description |
|--------|-------------|
| `upper_arm_mid_d_(cm)` | Глубина средней части верхней руки участников |
| `upper_arm_distal_d_(cm)` | Глубина дистальной части верхней руки участников |
| `upper_arm_proximal_d_(cm)` | Глубина проксимальной части верхней руки участников |
| `forearm_mid_d_(cm)` | Глубина средней части предплечья участников |
| `forearm_distal_d_(cm)` | Глубина дистальной части предплечья участников |
| `forearm_proximal_d_(cm)` | Глубина проксимальной части предплечья участников |
| `hand_d_(cm)` | Глубина кисти участников |


### EMG, IMU and PPG data folder: 

external - внешнее вращение, 
internal - внутреннее вращение, 

| Folder Name | Description |
|-------------|-------------|
| `30–40% внешнего поворота` | Упражнение на внешний поворот плеча с усилием 30–40 % от максимального добровольного усилия (MVIC). Содержит 34 папки с данными ЭМГ, ИМУ и ППГ. |
| `40–50% внешнего поворота` | Упражнение на внешний поворот плеча с усилием 40–50 % от максимального добровольного усилия (MVIC). Содержит 34 папки с данными ЭМГ, ИМУ и ППГ. |
| `50–60% внешнего вращения` | Упражнение на внешнее вращение плеча с усилием 50–60% от MVIC. Содержит 34 папки с данными ЭМГ, ИМУ и ППГ. |
| `30–40% внутреннего вращения` | Упражнение на внутреннее вращение плеча с усилием 30–40% от MVIC. Содержит 34 папки с данными ЭМГ, ИМУ и ППГ. |
| `40–50% внутреннего вращения` | Упражнение на внутреннее вращение плеча с усилием 40–50% от MVIC. Содержит 34 папки испытуемых с данными ЭМГ, ИМУ и ППГ. |
| `50–60% внутреннего вращения` | Упражнение на внутреннее вращение плеча с усилием 50–60% от MVIC. Содержит 34 папки испытуемых с данными ЭМГ, ИМУ и ППГ. |
| `Сила MVIC при внешнем вращении, первое измерение` | Первое измерение силы MVIC при внешнем вращении плеча. Данные PPG не включены. |
| `Вторая сила MVIC при внешнем вращении` | Второе измерение силы MVIC при внешнем вращении плеча. Данные PPG не включены. |
| `Первая сила MVIC при внутреннем вращении` | Первое измерение силы MVIC при внутреннем вращении плеча. Данные PPG не включены. |
| `Вторая сила MVIC при внутреннем вращении` | Второе измерение силы MVIC при внутреннем вращении плеча. Данные PPG не включены. |

**Subject Folder Structure (subject_x, x = 1-34):**

| Folder | Files | Description |
|--------|-------|-------------|
| `EMG data` | `anterior_deltoid.csv`, `infraspinatus.csv`, `latissimus_dorsi.csv`, `pectoralis_major.csv`, `posterior_deltoid.csv`, `upper_trapezius.csv` | Сигналы ЭМГ (мВ) для каждой мышцы |
| `IMU data` - `Forearm` | `acc_forearm.csv`, `gyr_forearm.csv`, `mag_forearm.csv` | Ускорение (м/с²), гироскоп (рад/с), магнитометр (а.е.) |
| `IMU data` - `Hand` | `acc_hand.csv`, `gyr_hand.csv`, `mag_hand.csv` | Ускорение (м/с²), гироскоп (рад/с), магнитометр (а.е.) |
| `IMU data` - `Pelvis` | `acc_pelvis.csv`, `gyr_pelvis.csv`, `mag_pelvis.csv` | Ускорение (м/с²), гироскоп (рад/с), магнитометр (а.е.) |
| `IMU data` - `Shoulder` | `acc_shoulder.csv`, `gyr_shoulder.csv`, `mag_shoulder.csv` | Ускорение (м/с²), гироскоп (рад/с), магнитометр (а.е.) |
| `IMU data` - `Sternum` | `acc_sternum.csv`, `gyr_sternum.csv`, `mag_sternum.csv` | Ускорение (м/с²), гироскоп (рад/с), магнитометр (а.е.) |
| `IMU data` - `Upper arm` | `acc_upper_arm.csv`, `gyr_upper_arm.csv`, `mag_upper_arm.csv` | Ускорение (м/с²), гироскоп (рад/с), магнитометр (а.е.) |
| `PPG data` | `ppg.csv` | Данные PPG (V) во время упражнения |


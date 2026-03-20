# FILES
- 'Stress_Level_v1.csv' and 'Stress_Level_v2.csv' contain self-reported stress levels during the stress protocol for each stage of the study,  respectively.
- 'subject-info.csv' contains demographic data such as age, weight, and height for all participants.
- 'data_constraints.txt' contains detailed information of issues such as incorrect wristband placements, incomplete protocols, and connection problems. 
- 'Wearable_Dataset.ipynb'  is a Jupyter Notebook that provides visualizations of all recorded signals and includes sample code for managing the data.
- The main folder 'Wearable_Dataset' contains three folders for each type of activity: 'STRESS', 'AEROBIC' and 'ANAEROBIC'.
- Each of these folders includes participant subfolders named as 'S01','S02','f01',etc.
- Each of the folders corresponding to each participant contains Empatica E4 files: 'ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'tags.csv' and 'TEMP.csv'.

# RELEVANT INFO
- Before downloading, we recommend reviewing the Jupyter Notebook ('Wearable_Dataset.ipynb') to determine whether the dataset fits your needs. This file is provided to read, open, visualize and start working with the data. To execute the notebook, ensure that basic Python libraries such as pandas,os, numpy, time, and matplotlib are installed.
- Participants from the first stage are labeled as 'Sxx' while those from the second stage are labeled as 'fxx'.
- Session dates and event marks are expressed in UTC and have been adjusted for de-identification. The time shifts have been applied consistently across all records to maintain signal alignment. 
- Empatica signal files: the first row is the initial time of the session expressed in UTC (Empatica provides time in Unix timestamp format, but files are already converted to UTC). The second row is the sample rate expressed in Hz.
- 'ACC.csv': data from x, y, and z axis are stored in first, second, and third column, respectively.
- 'IBI.csv': the first column is the time (respect to the initial time) of the detected inter-beat interval expressed in seconds (s). The second column is the duration in seconds (s) of the detected inter-beat interval (i.e., the distance in seconds from the previous beat).

---

# ФАЙЛЫ
- «Stress_Level_v1.csv» и «Stress_Level_v2.csv» содержат данные о самооценке уровня стресса во время стрессового протокола для каждого этапа исследования,  соответственно.
- «subject-info.csv» содержит демографические данные, такие как возраст, вес и рост всех участников.
- «data_constraints.txt» содержит подробную информацию о таких проблемах, как неправильное размещение браслетов, неполные протоколы и проблемы с подключением. 
- «Wearable_Dataset.ipynb»  — это Jupyter Notebook, который обеспечивает визуализацию всех записанных сигналов и включает пример кода для управления данными.
- Главная папка «Wearable_Dataset» содержит три папки для каждого типа деятельности: «STRESS», «AEROBIC» и «ANAEROBIC».
- Каждая из этих папок включает в себя подпапки участников с названиями «S01», «S02», «f01» и т. д.
- Каждая из папок, соответствующих каждому участнику, содержит файлы Empatica E4: «ACC.csv», «BVP.csv», «EDA.csv», «HR.csv», «IBI.csv», «tags.csv» и «TEMP.csv».


# СВЕДЕНИЯ
- Перед загрузкой рекомендуем ознакомиться с Jupyter Notebook («Wearable_Dataset.ipynb»), чтобы определить, соответствует ли набор данных вашим потребностям. Этот файл предоставляется для чтения, открытия, визуализации и начала работы с данными. Для запуска notebook необходимо убедиться, что установлены базовые библиотеки Python, такие как pandas, os, numpy, time и matplotlib.
- Участники первого этапа обозначены как «Sxx», а участники второго этапа — как «fxx».
- Даты сессий и отметки событий указаны в формате UTC и были скорректированы для деидентификации. Сдвиги во времени были применены ко всем записям для сохранения выравнивания сигналов. 
- Файлы сигналов Empatica: первая строка — начальное время сессии, выраженное в UTC (Empatica предоставляет время в формате Unix timestamp, но файлы уже преобразованы в UTC). Вторая строка — частота дискретизации, выраженная в Гц.
- «ACC.csv»: данные по осям x, y и z хранятся в первом, втором и третьем столбцах соответственно.
- «IBI.csv»: первый столбец — время (относительно начального времени) обнаруженного интервала между ударами, выраженное в секундах (с). Второй столбец — продолжительность в секундах (с) обнаруженного интервала между ударами (т. е. расстояние в секундах от предыдущего удара).


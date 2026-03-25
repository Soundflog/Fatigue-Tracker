# Multivariate Time Series data of Fatigued and Non-Fatigued Running from Inertial Measurement Units

## Ref

[Dataset on Zenodo](https://zenodo.org/records/7997851)

## Descrition

The data captured came from mounting a single Shimmer3 IMU on the lumbar of 19 recreational runners. The participants were all regular runners and injury free. The study protocol was reviewed and approved by the human research ethics committee at University College Dublin.

The data was collected in three segments; in the first, the participant completed a 400m run at a comfortable pace; the second segment consisted of a beep test which acted as the fatiguing protocol for this study; and the last segment where the runner was required to complete the 400m run at their comfortable pace, this time in their fatigued state. The beep test requires the runner to continuously run between two points 20m apart following an audio which produces `beeps` indicating when the person should begin running from one end to the other. The test eventually requires the runner to increase their pace as the interval between the `beeps` reduces as the test progresses. The fatiguing protocol ends when the runner is unable to keep up the increase in pace. The runs were all done on an outdoor running track. The sensor captured acceleration, angular velocity and magnetometer data throughout the three stages of the trials at a sampling rate of 256Hz. The data included here are segmented strides from the two 400m runs of each of the 19 participants. The labels on the data represent the participant number and whether it was a fatigued stride ('F') or a not fatigued stride ('NF').
The data used from the sensors includes data from the accelerometer in three directions (X, Y, Z) and the gyroscope in three directions (X, Y, Z). The direction of each of the axis is relative to the sensor. Two extra signals, magnitude acceleration and magnitude gyroscope were derived from the component signals and included in the analysis.

Kindly cite one of the following papers when using this data:

B. Kathirgamanathan, B. Caulfield and P. Cunningham, "Towards Globalised Models for Exercise Classification using Inertial Measurement Units," 2023 IEEE 19th International Conference on Body Sensor Networks (BSN), Boston, MA, USA, 2023, pp. 1–4, doi: 10.1109/BSN58485.2023.10331612

B. Kathirgamanathan, T. Nguyen, G. Ifrim, B. Caulfield, P. Cunningham. Explaining Fatigue in Runners using Time Series Analysis on Wearable Sensor Data, XKDD 2023: 5th International Workshop on eXplainable Knowledge Discovery in Data Mining, ECML PKDD, 2023, http://xkdd2023.isti.cnr.it/papers/223.pdf



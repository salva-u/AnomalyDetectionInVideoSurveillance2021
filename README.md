# AnomalyDetectionInVideoSurveillance2021
This Project aims at detecting anomalous events in Video Surveillance Footage. 


* [UCF-Crime Dataset](https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset) - CCTV Footage Dataset used.
* [R(2+1)D-152 Model](https://github.com/facebookresearch/VMZ) - The Residual(2+1)D-152 CNN from Facebook's VMZ Model Rep.

## Project Folders are as follows:
- C3D Code Files - Used to extract C3D Features to reproduce baseline work's results.
- VMZ Model Code - Used to extract Features using the R(2+1)D-152 Depth model.
- Multiple Instance Learning (MIL) Model Training and Testing Code 

### Prerequisites
This model was trained using CUDA 10.0 and GeForce RTX 2080 Ti GPU
config==0.5.0.post0\\
configparser==4.0.2\\
ffmpeg==1.4\\
h5py==2.10.0\\
imageio==2.6.1\\
Keras==2.3.1\\
Keras-Applications==1.0.8\\
Keras-Preprocessing==1.1.0\\
matplotlib==3.1.2\\
numpy==1.18.2\\
opencv-python==4.2.0.32\\
Pillow==5.1.0\\
PyOpenGL==3.1.0\\
PyQt5==5.13.1\\
PyQt5-sip==12.7.0\\
pyRFC3339==1.0\\
pytest==3.3.2\\
python-apt==1.6.5+ubuntu0.2\\
python-dateutil==2.8.1\\
python-debian==0.1.32\\
PyYAML==5.3\\
reportlab==3.4.0\\
scikit-image==0.16.2\\
scikit-learn==0.22.2.post1\\
scipy==1.4.1\\
sklearn==0.0\\
tensorboard==1.14.0\\
tensorflow==1.14.0\\
tensorflow-estimator==1.14.0\\
termcolor==1.1.0\\
Theano==1.0.4\\

## Baseline Work

* [Baseline Paper and Work]( https://www.crcv.ucf.edu/projects/real-world/) - Baseline Conference Paper
* [Baseline Works Code](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018) - The original code used in this rep, heavily borrowed from here.

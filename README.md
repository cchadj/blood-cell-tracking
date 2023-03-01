# Towards a Deep Learning Pipeline for Measuring Retinal Bloodflow

# Abstract

Alterations to microvascular flow are responsible for a number of ocular and
systemic conditions including diabetes, dementia and multiple sclerosis. The challenge is developing of methods to capture and quantify retinal capillary flow in the
human eye. We built a bespoke adaptive optics scanning laser ophthalmoscope with
two spatially offset detection channels, with configurable offset aperture detection
schemes to image microvascular flow. In this research we sought to develop an automatic tool that detects and tracks erythrocytes. A deep learning convolutional neural
network is proposed for classifying blood-cell from non-blood-cell patches in each
frame. The patch classification is coupled with a localisation process to detect the
positions of the red blood cells. A capillary segmentation method is also presented
to increase the efficiency and performance of the localisation process. Finally, a
technique is presented to match corresponding cells between the two channels in
the raster, allowing for a fully automatic blood flow velocity measurement. Results
from various experiments are reported and compared to give the most accurate configuration. The deep learning basis of the tool allows for a continual and adaptable
learning that can improve the performance of the tool as more samples are collected
from subjects. In addition, the modular nature of the tool allows for replacing its
components, such as the capillary segmentation, with state-of-the-art techniques
allowing for the software’s longevity.

# Thesis PDF

[Towards a Deep Learning Pipeline
for Measuring Retinal Bloodflow PDF](https://github.com/cchadj/blood-cell-tracking/files/10862213/Chrysostomos_Chadjiminas___blood_cell_tracking___thesis_report.pdf)

# Blood Cell Classification

For blood cell classification we train a CNN with positives and negatives patches extracted from the training videos.</br>
Positive patches are centered around erythrocytes while negative patches are patches extracted arround but not centered on erythrocytes.</br>
At inference time we extract a patch arround each pixel of the image and assign a probability of it being an erythrocite based on the output of the CNN.</br>
This process produces a probability map from which the locations of the cells are estimated. 

For more info please refer to the **3.2** section in the [thesis pdf](https://github.com/cchadj/blood-cell-tracking/files/10862213/Chrysostomos_Chadjiminas___blood_cell_tracking___thesis_report.pdf).

Here is an example of a registered input video of the retina on the right and the output probability map on the left.
In the probability map the blue dots centered around the probability blobs signify the estimated erythrocyte location.

Input Registered Video     |  Probability Map with estimated cell locations as blue dots
:-------------------------:|:-------------------------:
![Input Registered Video](https://user-images.githubusercontent.com/22410337/222170570-3df557d3-ab71-488d-b0f4-04620532edf8.gif)  |  ![Output Estimations ( Over Probability Map ) ](https://user-images.githubusercontent.com/22410337/222170449-b42c50a4-85bb-4987-98be-064da3a44039.gif)

# Vesselness - Capillary detection
To improve the accuracy of our estimation and also optimise the process of estimating the locations of the erythrocytes for each frame of the input we must reduce the search-space to the capillaries of the retina as thhere can't be erythrocytes outside of the capillaries.</br>
To improve this we extract a vessel mask for each video by applying an image processing pipeline to the standard deviation image of the video.

For more info please refer to the **3.4** section in the [thesis pdf](https://github.com/cchadj/blood-cell-tracking/files/10862213/Chrysostomos_Chadjiminas___blood_cell_tracking___thesis_report.pdf).

Standar Deviation Image | Output Vessel Mask
:-------------------------:|:-------------------------:
![Subject10_Session109_OD__10_1x1_416_OA790nm1_extract_reg_cro](https://user-images.githubusercontent.com/22410337/222188269-30696636-f295-451f-86d4-aeafde2bfd13.png) | ![Vessel Mask](https://user-images.githubusercontent.com/22410337/222188522-e6e4dfcf-a3d5-40ce-805b-764ae3e900f6.png)


### Set up
1. Download data from : https://liveuclac-my.sharepoint.com/personal/smgxadu_ucl_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsmgxadu%5Fucl%5Fac%5Fuk%2FDocuments%2FShared%5FVideos&ct=1583323140391&or=OWA-NT&cid=9c7726fb-db68-e102-a4a7-f93127374108&originalPath=aHR0cHM6Ly9saXZldWNsYWMtbXkuc2hhcmVwb2ludC5jb20vOmY6L2cvcGVyc29uYWwvc21neGFkdV91Y2xfYWNfdWsvRWx0WXpFMFBVWHREc1RBY0NoQk5TY1lCSllSV2dkQmVOMmZuWHNZSmhCZ1BDQT9ydGltZT0wdG1lYXpQQTEwZw
2. Extract data to Shared_Videos folder

### Demos:
0. - PATCH EXTRACTION (shows patch extraction methods)

1. - VESSEL MASK CREATION (demonstrates the vessel mask creation for the validation videos)
3. - AVERAGE CELL MATCHING
4. - UID6-validation (shows the results on running model with unique id 6 on the validation data. shows the worst and best probability maps
for each validation video)
5. - CHANNEL REGISTRATION (shows the channel registration results)
"# blood-cell-tracking" 

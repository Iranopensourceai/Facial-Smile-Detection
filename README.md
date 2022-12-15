<h1 align="center">Facial Smile Detection</h1>
<p align="center" width="100%">
    <img width="40%" src="https://github.com/Iranopensourceai/Facial-Smile-Detection/blob/main/figs/Smile.jpg">
</p>       

## Table of contents
- [Overview](#Overview)
- [Datasets](#Datasets)
- [Preprocessing](#Preprocessing)
    - [image preprocessing](#image_preprocessing)
    - [image augmentation](#image_augmentation)
- [Model](#Model)
    - [Building the model](#Building_model)
    - [Training](#Training)
- [Results](#Results)

## Overview     <a name="Overview"></a>
What makes a face more excellent? It could be a smile. So, in this project, we have worked on Smile detection using CNN & model sequential to train the model.
**Requirements:**
Before going deeper, you must install Python +3.6 & the following packages to run the code:

- CV2: pip install OpenCV-python
- Matplotlib: pip install matplotlib
- Tensorflow: pip install TensorFlow
- Sklearn: pip install scikit-learn

## Datasets     <a name="Datasets"></a>

## Preprocessing     <a name="Preprocessing"></a>
- ### image preprocessing      <a name="image_preprocessing"></a>
- ### image augmentation       <a name="image_augmentation"></a>


## Model       <a name="Model"></a>
<img align="left" width="33%" src="https://github.com/Iranopensourceai/Facial-Smile-Detection/blob/main/figs/model.JPG">     
     
### Building the model     <a name="Building_model"></a>
The CNN that is composed of:

- Conv2D layer with 32 filters, a kernel size of (3, 3), the relu activation function, a padding equal to same and the correct input_shape    
- MaxPooling2D layer with a pool size of (2, 2)    
- Conv2D layer with 64 filters, a kernel size of (3, 3), the relu activation function, and a padding equal to same    
- MaxPooling2D layer with a pool size of (2, 2)   
- Conv2D layer with 128 filters, a kernel size of (3, 3), the relu activation function, and a padding equal to same   
- MaxPooling2D layer with a pool size of (3, 3)   
- Flatten layer   
- dense function with 50 neurons with the relu activation function   
- dropout layer (with a rate of 0.5), to regularize the network   
- dense function related to the task: binary classification > sigmoid   


### Training            <a name="Training"></a>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Results              <a name="Results"></a>

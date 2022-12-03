<h1 align="center">Facial Smile Detection</h1>
<p align="center" width="100%">
    <img width="40%" src="https://github.com/Iranopensourceai/Facial-Smile-Detection/blob/main/figs/Smile.jpg">
</p>

## data collection 
dataCollection group : we downloaded dataset from kaggle.com
this data set have Smile,NoSmile,Test Folders 

## train test split

## Data pre-processing and data augmentation
The first version of preprocessing, spliting and agumentation on FirstDataSet commited.

## data visualization

## Model
<img align="left" width="33%" src="https://github.com/Iranopensourceai/Facial-Smile-Detection/blob/main/figs/model.JPG">     
     
### Building the model
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


### Training and visualizing the results
        
        
        
 

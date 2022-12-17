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
- [Built with](#Built_with)
- [Usage](#Usage)
- [Results](#Results)
- [Conclusion](#Conclusion)
- [Contributing](#Contributing)
- [References](#References)

## Overview     <a name="Overview"></a>

## Datasets     <a name="Datasets"></a>

## Preprocessing     <a name="Preprocessing"></a>
[It is a data mining technique that transforms raw data into an understandable format. Raw data (real world data) is always incomplete and that data cannot be sent through a model. That would cause certain errors. That is why we need to preprocess data before sending through a model.
Steps in Data Preprocessing
Here are the steps we have followed;
1. Import libraries
2. Read data
3.  Augmentations.
4.  for more information please check this link(Data Preprocessin-arash.docx)]

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

## Built with           <a name="Built_with"></a>
* [![Python][Python.org]][Python-url] 3.8.16 
* [![Tensorflow][Tensorflow.org]][Tensorflow-url] 2.9.2
* [![Keras][Keras.io]][Keras-url] 2.9.0
* [![Opencv][Opencv.org]][Opencv-url]  4.6.0
* [![Numpy][numpy.org]][numpy-url] 1.21.6
        
        
        
## Usage                <a name="Usage"></a>
 
First, scrap dataset with:
```
python web-scraping.py 'data/urls.csv' 'data/haarcascade_frontalface_default.xml'
```
To train the model with scraped dataset:
```
python train.py 'Dataset' 100 -w 128 -e 128
```
To test with an existing model:
```
python test.py 'jadi.jpg' -w 128 -e 128
```



## Results              <a name="Results"></a>
The model is trained on the mentioned dataset and the results are as follows.

### Loss-Accuracy plots     <a name="Loss-Accuracy plots"></a>
The plots show the (train - validation Accuracy) and (train - validation Loss) under 60 epochs.

<p align="center">
<img align="center" width="33%" src="https://github.com/Iranopensourceai/Facial-Smile-Detection/blob/main/figs/plots.png">     
</p>

### Evaluation metrics     <a name="Evaluation metrics"></a>
Metrics calculated to evaluate the model:

*Confusion matrix:*

`metrics.confusion_matrix(test_y, predictions)`


||Predicted(nonsmile)|predicted(smile)|
|:---:|:---:|:---:|
|**Actual(nonsmile)**|66|5|
|**Actual(smile)**|4|109|

*Classification report:*

`classification_report(test_y, predictions, target_names=['non smile','smile'])`

||precision |recall|f1-score |support|
|:---:|:---:|:---:|:---:|:---:|
|non smile|0.94|0.93 |0.94|71|
|smile|0.96|0.96|0.96|113|
|total||||184|

*Test evaluation:*

`model.evaluate(test_X,test_y)`

||Test|
|:---:|:---:|
|accuracy|0.9511|
|loss|0.1484|

An explanation of some metrics for evaluating classification models is provided [here](https://github.com/Iranopensourceai/Facial-Smile-Detection/files/10158556/classification_evaluation_metrics.pdf).

### Test on an image     <a name="Test on an image"></a>

Check the accuracy of smile detection:

<p align="center">
<img align="center" width="33%" src="https://github.com/Iranopensourceai/Facial-Smile-Detection/blob/main/figs/test_smile.jpg">     
</p>

Check the accuracy of non smile detection:

<p align="center">
<img align="center" width="33%" src="https://github.com/Iranopensourceai/Facial-Smile-Detection/blob/main/figs/test_nonsmile.jpg">     
</p>

## Conclusion       <a name="Contributing"></a>
As seen in plots and numerical results, the model has a good performance (Test accuracy = 95%). Generality and overfitting avoidance are significant features of the model, which are mainly derived from implementing the augmentation layer.

## Contributing       <a name="Contributing"></a>
### Issues
Issues are very valuable to this project. You can simply open an issue with the tag "enhancement".

* Ideas are a valuable source of contributions others can make
* Problems show where this project is lacking
* With a question you show where contributors can improve the user experience

### Pull Requests
If you have a suggestion that would make this better, please fork the repo and create a pull request. 

1. Fork the Project
2. Create your feature Branch (git checkout -b 'branch_name')
3. Commit your changes (git commit -m 'Add feature')
4. Push to the Branch (git push origin 'branch_name')
5. Open a Pull Request

Thank you for your contribution!

## References             <a name="References"></a>
[[1.]](https://arxiv.org/abs/1409.1556) Very Deep Convolutional Networks for Large-Scale Image Recognition

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python.org]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue
[Python-url]: https://python.org/
[Tensorflow.org]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[Tensorflow-url]: tensorflow.org/
[Keras.io]: https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white
[Keras-url]: https://keras.io/
[Opencv.org]: https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white
[Opencv-url]: https://opencv.org/
[numpy.org]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/

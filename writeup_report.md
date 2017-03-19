#**Behavioral Cloning** 

##Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_sample.jpg "Center Sample"
[image2]: ./examples/center_reversed_sample.jpg "Center Reversded Sample"
[image3]: ./examples/center_image_sample2.png "Center Image Sample 2"
[image4]: ./examples/center_image_sample2_flipped.png "Center Image Sample 2 Flipped"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing the vehicle driving in autonomous mode around the track using the trained model

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the proposed NVIDIA Network, which consists of a input layer (for images with dimensions 320x160) where data is normalized using a Keras lambda layer, a cropping layer, where I remove the top and botton sections of the image - those sections where the sky and trees and the hood of the car, which is unecessary for the model (model.py lines 68-69). The model is followed with 4 convolutional networks with 5x5 and 3x3 filters and depths between 24 and 64 with a RELU activation (model.py lines 70-74) and by 4 fully connected layers (model.py lines 75-79).

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 16, 58-59 and 82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

The model also uses a correction factor for side camera images. The correction factor is set to 0.38, so for any side camera images, the steering measurement will be adjusted to the original value * 0.38 (for left cameras) or the original value * (-0.38) (for right cameras). That is done because some of the input data collected from the simulator contained a steering measurement of zero even when the car was performing a turn.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (3 laps around the track) and reverse driving (driving the track clockwise - 2 laps). 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to derive a model that could scale well with big amounts of data as well as perform a good job on keeping the car on the track when on autonomous mode.

My first step was to use the Network provided on the lectures describing the model with center images only. The Network performed reasonably good, but it could not scale - with a small set of data, the network was able to keep the car on the track for a few turns. However, when I tried to use more data to increase the performance the network wasn't able to handle it. The network consisted a Keras lamba layer, a cropping layer, and 2 convolution and max-pooling layers.

In order to scale the Network I decided to use Keras generators. With the generator in place, the network was able to handle big amounts of data, but that didn't improve it enough to make it work. 

I then decided to try the NVIDIA Network. After implementing it and adjusting it to the correct input and normalization data, I was able to get much better results, with the car staying on track for much longer, but that was still not totally satisfactory. I then decided to use the left and right images to improve the network. I was able to get much better results but still not sufficient to complete a total lap around the track. Next I decided to augment the data by adding the fliped images/measurements to the array of training data. 

With that in place the results improved considerably. Next I applied a correction factor to the side cameras measurements, as most of the side cameras steering measurements were set to zero. I played around with different values until I found a correction value that provided the best results.

To combat the overfitting, I use the "Adam" optimizer to adjust the learning rate. I also split the data between training and validation sets.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 67-80) consisted of a convolution neural network with the following layers and layer sizes: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keras lambda - Input         		| 160x320x3, normalized image   							| 
| Keras Image Cropping         		| Crops top by 70px and bottom by 25px   							| 
| Convolution 5x5     	| 2x2 stride, outputs 78x158x24 with RELU activation 	|
| Convolution 5x5     	| 2x2 stride, outputs 37x77x36 with RELU activation 	|
| Convolution 5x5     	| 2x2 stride, outputs 17x37x48 with RELU activation 	|
| Convolution 3x3     	| 1x1 stride, outputs 15x35x64 with RELU activation 	|
| Convolution 3x3     	| 1x1 stride, outputs 13x33x64 with RELU activation 	|
| Fully connected					|		outputs 100										|
| Fully connected					|		outputs 50										|
| Fully connected					|		outputs 10										|
| Fully connected					|		outputs 1									|

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded 2 laps of the vehicle driving clockwise around the track (reversed mode, as the track is actually counter clockwise). That would allow the model to not be too bias toward left turns as most of the track is composed by left turns.

![alt text][image2]

To augment the data set, I also flipped images and angles thinking that this would help the model to not be too bias toward left turns. For example, here is an image that has then been flipped:

Original Image:
![alt text][image3]

Flipped Image:
![alt text][image4]


After the collection process, I had 5712 number of data points. I then preprocessed this data by normalizing each image and by cropping the top and bottom of the images - which represents the sky/trees and the hood of the car respectively. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as after the 5th epoch both the validation and training loss accuracy would start increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.

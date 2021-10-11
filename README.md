# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./images/crop.png "Cropped image"
[image2]: ./images/center1.jpg "Center cam"
[image2b]: ./images/left1.jpg "Left cam"
[image2c]: ./images/right1.jpg "Right cam"
[image3]: ./images/save1.jpg "Recovery Image"
[image4]: ./images/save2.jpg "Recovery Image"
[image5]: ./images/save3.jpg "Recovery Image"
[image6]: ./images/flip1.jpg "Normal Image"
[image7]: ./images/flip2.jpg "Flipped Image"


---
### Files 
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
and then running the simulator in autonomous mode (on Track 1)


The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture 

My model consists of a convolution neural network with 6 filters with 5x5 kernel sizes. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 


The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 77).


Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving the track in reverse (to reduce the left bias).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Design Approach


My first step was to use a convolution neural network model similar to the LeNet architecture that we used in the Classifying Signs Project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80-20 split). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it was cropping out 50 pixels from the top and 20 from the bottom. This removed any useless information from the model's training. Then I normalized the image.

![alt text][image1]

The final step was to run the simulator to see how well the car was driving around track one. Since, I had spent a meticulous time collecting the training data, there were no spots where the vehicle drove off the track and actually recovered well when it was about to drift off (see run1.mp4). 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   			| 
| Cropping2D     	| 50 rows pixels from the top of the image, 20 rows pixels from the bottom of the image 	|
| Lambda | Normalizing the image |
| Convolution 5x5	    |  Valid padding|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Valid padding		    |
| Convolution 5x5	    |  Valid padding|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Valid padding		    |
| Flatten	        	| 				                    |
| Fully connected		| Outputs 120        							|
| Fully connected		| Outputs 84        							|
| Fully connected		| Outputs 1 (Regression)      				|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]
![alt text][image2b]
![alt text][image2c]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the center if it drifted off course. These images show what a recovery looks like... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process by driving in reverse on the track to reduce the left-steering bias of track 1. I drove off road and then recorded myself driving back to the center.

To augment the data sat, I also flipped images and angles thinking that this would give me a bigger dataset. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 32000+ number of data points. I then preprocessed this data by normalizing it and cropping it to cut out the useless information. I did not need a generator in my code to process these images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by trial and error. I used an adam optimizer so that manually training the learning rate wasn't necessary.

# **Traffic Sign Recognition**

## Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[graphs]: ./graphs.png "Graphs"
[acts]: ./acts.png "Feature maps"
[image0]: ./signs_from_web/0.png "Traffic Sign 1"
[image1]: ./signs_from_web/1.png "Traffic Sign 2"
[image2]: ./signs_from_web/2.png "Traffic Sign 3"
[image3]: ./signs_from_web/3.png "Traffic Sign 4"
[image4]: ./signs_from_web/4.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the standard python library methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The below is a tow bar charts showing how many image data each class contains, and their ratio.

![Graphs of data][graphs]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing I've decided to do only one thing – normalize all image data as suggested Andrej Karpathy in his cs231n course:

`(channel - channel_mean) / channel_std`

It gives me image data with channels mean near zero and with stddev near 1.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I've decided not to take LeNet model as there are more simplier and effective atrchitectures around.
I've decided to make my own small vgg-like net, but with smaller amount of layers as task is pretty simple.
My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x16, relu, BatchNorm 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x16, relu, BatchNorm 	|
| Maxpool 2x2           | 2x2 stride, valid padding |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x32, relu, BatchNorm 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x32, relu, BatchNorm 	|
| Maxpool 2x2           | 2x2 stride, valid padding |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x64, relu, BatchNorm 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x64, relu, BatchNorm 	|
| Maxpool 2x2           | 2x2 stride, valid padding |
| Flatten	              | |
| Fully connected		    | inputs 256, outputs 43 (num classes), linear 									|
| Softmax			        	| |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For training I've used default settings for Adam optimizer. I've added early stopping after two epochs without improvement.
And saved checkpoint of best validation result

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.95+
* validation set accuracy of 0.978
* test set accuracy of 0.962

If an iterative approach was chosen:
At first I've made less layers with bigger depth of convolution layers. But on such small images better works not so big depth in beginning.
I've got result more than 0.93 from first try, but to improve I've decided to add batchnorm layers after relu, as it shows very good results on my submissions for kaggle.
I've tuned amount of epochs, as at first I've used 10 epochs, but with early stopping algorithm you can set it any big number you want, it will stop when it finds good solution. So I've changed it to 100. I've started with batch size 32, but then changed it to 64.
Also it is not very well known fact that relu should go before batchnorm because in original work author of batchnorm used it in other order. But after that work author and other scientists agreed that relu->BN works better in most cases [link to reddit](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image0] ![alt text][image1] ![alt text][image2]
![alt text][image3] ![alt text][image4]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Go straight or left      		| Go straight or left   									|
| Yield					| Yield											|
| General caution	      		| General caution					 				|
| No passing vechiles over 3.5 metric tons      		| No passing vechiles over 3.5 metric tons		 				|
| Priority road	      		| Priority road					 				|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Here are top-5 predictions for my first three images:

Correct label:  Go straight or left
Best match:  Go straight or left  :  0.999098
Best match:  Ahead only  :  0.000880729
Best match:  Priority road  :  1.37293e-05
Best match:  Slippery road  :  1.99933e-06
Best match:  Speed limit (120km/h)  :  1.93926e-06

Correct label:  Yield
Best match:  Yield  :  0.997268
Best match:  No passing  :  0.00272196
Best match:  Speed limit (100km/h)  :  4.88005e-06
Best match:  Speed limit (120km/h)  :  2.69296e-06
Best match:  No vehicles  :  1.58414e-06

Correct label:  General caution
Best match:  General caution  :  1.0
Best match:  Speed limit (30km/h)  :  2.66061e-08
Best match:  Traffic signals  :  3.68164e-09
Best match:  No vehicles  :  1.30139e-09
Best match:  Pedestrians  :  1.59156e-10

For other 4 images on which I've did my tests probability of right label looks the same – near 100%.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I've done visual output of featured maps in last cell of my notebook.

![Feature maps][acts]


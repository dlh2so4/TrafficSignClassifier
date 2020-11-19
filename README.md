# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./Writeup/bar.PNG "Visualization"
[image2]: ./Writeup/grayscale.PNG "Grayscaling"
[image4]: ./TestImages/a.jpg "Traffic Sign 1"
[image5]: ./TestImages/b.jpg "Traffic Sign 2"
[image6]: ./TestImages/c.jpg "Traffic Sign 3"
[image7]: ./TestImages/d.jpg "Traffic Sign 4"
[image8]: ./TestImages/e.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
    34799
* The size of the validation set is ?
    4410
* The size of test set is ?
    12630
* The shape of a traffic sign image is ?
    (32, 32, 3)
* The number of unique classes/labels in the data set is ?
    43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes among the 43 classes

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because of 3 reasons:
    1. For the traffic sign, even with grayscale iamges, meaning can be determined.
    2. Grayscale image only has 1 color channel, which will save memory compare with 3 color channel images.
    3. After a test, I found that the preprocess of grayscale can improve accuracy in the result. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because this will be able to improve the accuracy of the result(I tried with and without normalized image, the accuracy can improve a lot when normalize the images.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					|     Description								| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 Grayscale image						| 
| Convolution 28x28		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6 				|
| Convolution 10x10		| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16 				|
| Flatten				| outputs 400      									|
| Fully connected		| outputs 120      									|
| RELU					|												|
| dropout				|												|
| Fully connected		| outputs 84      									|
| RELU					|												|
| dropout				|												|
| Fully connected		| outputs 43      									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with 128 batch size, 60 epochs and 0.001 learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
    0.999
* validation set accuracy of ? 
    0.958
* test set accuracy of ?
    0.947
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
The LeNet Model Architecture was chosen.
* Why did you believe it would be relevant to the traffic sign application?
The LeNet Architecture is well known for classification of images, it meets the goal of the traffic sign classifier.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The training set, validation set and test set all have more than 0.947 accuracy, this is a pretty good result. It shows that the model is able to give good prediction.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The 4th image might be difficult to classify because it has shadow on one side. The 5th image is blur, so it can potentially be another challenge. When I run the preprocess without grayscale and normalization, they did not get recognized. They got successfully predicted when I increase grayscale and normalization.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					| Prediction									| 
|:---------------------:|:---------------------------------------------:| 
| General caution 		| General caution								| 
| Children crossing		| Children crossing								|
| Slippery road			| Slippery road 								|
| Ahead only			| Ahead only									|
| No entry				| No entry					 					|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.947.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a General caution sign (probability of 100%), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability 			|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| 100%					| 18-General caution   									| 
| 0%					| 26 										|
| 0%					| 14											|
| 0%					| 25					 				|
| 0%					| 17      							|


For the first image, the model is relatively sure that this is a General caution sign (probability of 100%), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability 			|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| 100%					| 17-No entry   									| 
| 0%					| 0 										|
| 0%					| 1											|
| 0%					| 26					 				|
| 0%					| 40      							|


For the first image, the model is relatively sure that this is a General caution sign (probability of 100%), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability 			|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| 85%					| 23-Slippery road   									| 
| 15%					| 29 										|
| 0%					| 9											|
| 0%					| 19					 				|
| 0%					| 35      							|


For the first image, the model is relatively sure that this is a General caution sign (probability of 100%), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability 			|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| 94%					| 28-Children crossing   									| 
| 4%					| 30 										|
| 2%					| 34											|
| 0%					| 29					 				|
| 0%					| 18      							|


For the first image, the model is relatively sure that this is a General caution sign (probability of 100%), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability 			|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| 100%					| 35-Ahead only   									| 
| 0%					| 3 										|
| 0%					| 15											|
| 0%					| 9					 				|
| 0%					| 10      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



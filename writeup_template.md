# **Traffic Sign Recognition** 

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/preconv_fig.png "Preprocessing network"
[image3]: ./examples/random_transform.jpg "Random Transform"
[image4]: ./examples/convnet_fig.png "Model Architecture"
[image5]: https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg "Dense block"
[image6]: ./examples/placeholder.png "Traffic Sign 1"
[image7]: ./examples/placeholder.png "Traffic Sign 2"
[image8]: ./examples/placeholder.png "Traffic Sign 3"
[image9]: ./examples/placeholder.png "Traffic Sign 4"
[image10]: ./examples/placeholder.png "Traffic Sign 5"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training samples are distributed among each traffic sign class.

![visualization.jpg][image1]

As you can see the number of training samples of each class varies widely. (I have explained below the steps I have taken to make it more balanced among different classes at preprocessing step.)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a preprocessing step, I normalized the image data to range of [0,1] using min-max scaling because having large pixel values in the range of [0,255] could cause the network to get saturated and making hard to train the network.

I did not used any other preprocessing like grayscalling or histogram normalization etc. Instead as proposed in ["Systematic evaluation of CNN advances on the ImageNet"](https://arxiv.org/pdf/1606.02228.pdf), I used a mini network at the beginning of my model to learn a transformation via convolution. While original paper suggests using 1x1 convolutions, I used 3x3 and 1x1 convolutions respectively at this step which gave better results.

I decided to generate additional data because it helps to prevent model from overfitting on training set since samples were not equally distributed between each classes in the original training set. 

To add more data to the data set, I used the following techniques because they would generate as much as legitimate samples without introducing unrealistic image effects to the training samples. 

First I used flipping of existing samples of symmetrical signs wherever possible. For this I used the flipping method available at, https://navoshta.com/traffic-signs-classification/#flipping 

Then I balanced the data between classes by cloning existing samples so that there are at least 2000 samples per each class. At each iteration of training a randomly selected portion of these were perturbed using random translations, rotations and perspective transformations. In this step I also used border reflect method to prevent introducing high contrast unwanted features and keep the border areas consistent with original images. The basic idea of this approach is to create "infinite" augmented training data set by generating new data during the training epoch. When training data is always changing, the network will be less prone to overfitting. However at the same time it is necessary to make sure that the data change is not too big so that it won't cause "jumps" in the training loss.

Here is an example of an original image and an augmented image:

![random_transform.jpg][image3]

The difference between the sample counts in original data set and the augmented data set is the following.

* The size of extended training set is 112134
* Minimum number of samples per class in original data set is 180
* Maximum number of samples per class in original data set is 2010
* Minimum number of samples per class in extended data set is 2010
* Maximum number of samples per class in extended data set is 7560

Note that due to few classes that had high number of symmetrical traffic signs there were few outliers that got high number of samples than average of 2000 samples after this operation.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
|      	| Preprocessing Mini-Net 	|
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x10 	|
| Batch normalization     	|  	|
| VLReLU (alpha=0.25)					|												|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x3 	|
| Batch normalization     	|  	|
| VLReLU (alpha=0.25)					|												|
|      	| Densenet Model 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| Dense block	      	| growth rate:24, 4 layers, outputs 32x32x128 				|
| Batch normalization     	|  	|
| ReLU					|												|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x128 	|
| Dropout     	| Keep probability : 0.8	|
| Avg pooling     	| 2x2 stride,  outputs 16x16x128 	|
| Dense block	      	| growth rate:24, 4 layers,  outputs 16x16x224 				|
| Batch normalization     	|  	|
| ReLU					|												|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 16x16x224 	|
| Dropout     	| Keep probability : 0.8	|
| Avg pooling     	| 2x2 stride,  outputs 8x8x224 	|
| Dense block	      	| growth rate:24, 4 layers,  outputs 8x8x320 				|
| Batch normalization     	|  	|
| ReLU					|												|
| Avg pooling     	| 8x8 stride,  outputs 1x1x320 	|
| Fully connected		| 43 output units        									|
| Softmax				|         									|

Instead of any hand tuned image preprocessing step, I have used a Mini-network for learning color space transformations as described in https://arxiv.org/pdf/1606.02228.pdf . One noticeable exception is that while original paper suggests using two 1x1 convolutions consecutively, I have instead used 3x3 and 1x1 convolutions respectively.

Following diagram shows the model architecture of Mini-Net used for learning colorspace transformations at the beginning of the model.
![preconv_fig.png][image2]

This was followed by a DenseNet model as proposed by Gao Huang et al in https://arxiv.org/pdf/1608.06993.pdf However I have reduced the number of layers in each dense block to 4 and growth rate to 24 to make it a more manageable sized network.

Following diagram shows the model architecture of DenseNet portion of the model.
![convnet_fig.png][image4]

(These figures are generated by adapting the code from https://github.com/gwding/draw_convnet)

In this model a dense block contains several layers of convolutions where each layer is directly connected to every other layer in a feed-forward fashion. For each layer, the feature maps of all preceding layers are treated as separate inputs whereas its own feature maps are passed on as inputs to all subsequent layers.

Following diagram shows construction of a dense block with 5 layers and growth rate 4.
![fig1.jpg][image5]
(Image source: https://github.com/liuzhuang13/DenseNet)

For my network I have used dense blocks with 4 layers and growth rate of 24.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a MomentumOptimizer with momentum 0.9 and Nesterov Momentum. Final hyper parameter settings that was used are:

* learning rate: 0.05, 0.02, 0.005, 0.002, 0.0005 and 0.0002 at epoch 1, 3, 6, 8, 11 and 16 respectively.
* batch size: 16
* epochs: 20
* L2 normalization beta: 1e-3
* dropout layers keep probability: 0.80
* random training sample generator keep probability: 0.40

While I used MomentumOptimizer with ```momentum=0.9``` and truncated normal initializer with ```mu``` and ```sigma``` for weights. I did not tuned these hyperparameters for brevity.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.998
* test set accuracy of 0.9960

I have chosen a systematic iterative approach to finding a solution for getting a validation set accuracy above 0.93.

For this first I had chosen the LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson. This was chosen as it is used to solve similar kind of classification problem and input sizes matches after either adjusting for the color depth of 3 channels or preprocessing input images to grayscale.

However this architecture could only achieve 0.918 accuracy of validation set after global histogram equalization.

So I evaluated the performace of Sermanet LeCun model as described in http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf but with comparably similar network size. This increased validation set accuracy to 0.927. Then I increased model complexity by increasing number of features to 22 and 38 in first and second convolutional layer respectively. I also used additional hidden layer of size 120 between flattened and output layers of the network. This achieved 0.94 validation accuracy. These changes helped to prevent the network from underfitting by increasing training set accuracy from below 0.986 to 0.998. However at this stage the network started overfitting the training data.

While I was searching for measures to prevent overfitting, my mentor suggested LeNet-5 based network with following parameters.  
CNN Layers: 4 Hidden (2 CNN + 2 MatMul)  
Layer 1: Convolutional + MaxPool. The output is 14 x 14 x 16 Activation : RELU  
Layer 2: Convolutional + Max Pool. The output is 5 x 5 x 16  
Layer 3: Matmul. The output is 256 x 64  
Layer 4: Matmul. The output is 64 x 43  
Sizes: batch size = 16  
conv filter patch size = 5  
conv filter depth = 16  
Optimizer: Adam Optimizer with learning rate of 0.001  
While I have tried similar networks before moving into Sermanet LeCun model, what I observed here is that reducing batch size from 128 to 16 could increase the validation set accuracy for same model in this case. And increasing the feature size of convolutional layers to 16, 40 respectively on this model I could obtain 0.951 validation accuracy.

To prevent networks from overfitting I introduced dropout and L2 regularization and I tested these approaches in both Sermanet and LeNet-5 architectures of moderate sizes. Dropout layers helps network to generalize by preventing the model from being too much relying on a particular feature at hidden layers while L2 regularization penalizes very large weights (And thus again preventing model being too inclined for particular features) This gives me validation accuracy of 0.966 and 0.973 for LeNet-5 and Sermanet/LeCun models respectively.

While doing these tuning of LeNet-5 and Sermanet models I kept exploring more literature on deep learning based image classifications. Based on the paper at https://arxiv.org/pdf/1606.02228.pdf I tried out using a mini network at the beginning of the model to learn the color space transformations instead of using a image preprocessing step to normalize the color validations between images. This yielded 0.971 accuracy with LeNet-5 model - on par or better than using histogram equalized image preprocessing.

Then I employed batch normalization to further reduce overfitting. This yielded 0.976, 0.983 and 0.976 for LeNet-5, LeNet-5 with Mini-Net and Sermanet models respectively.

Further readings introduced me to DenseNet model which is an state of the art network proposed recently[[2]](https://arxiv.org/pdf/1608.06993.pdf). Even at first attempt of using moderate size DenseNet employing 4 layers with growth rate of 12 at each block resulted in 0.992 validation accuracy with histogram equalization for image preprocessing. However combining DenseNet with Mini-Net for learning colorspace transformations did not yield results on par with histogram equalization based image preprocessing for same model. But after tuning the models a bit, it was on par with or better than LeNet-5 based models.

To further reduce overfitting I decided to generate additional data as described above. This increased the validation accuracy of all the models. At this point I used CLAHE for any models that were using histogram equalization for image preprocessing. After these changes following were the standings of each model.  
LeNet-5 with CLAHE 0.985  
Sermanet with CLAHE 0.982  
DenseNet with CLAHE 0.993  
LeNet-5 with Mini-Net (Transnet+LeNet) 0.988  
DenseNet with Mini-Net (Transnet+DenseNet) 0.998

Then I tuned following hyper parameters individually in a similar fashion to problem 3 in [TensorFlow lab](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/lab.ipynb). Here I would select a parameter to tune and keeping all other parameters constant I would train the network and choose the value that gives the best accuracy. Ranges of parameters used for tuning are as follows. To reduce the computing time for this I used original training data set instead balanced data set. However at each epoch a randomly selected portion of the training data set was randomly transformed to generate new samples.
```
perturb_rates = [0.8,0.6,0.4,0.2] # 1.0-keep_radio for random perturbations of train set.
batch_sizes = [16,32,64,128]
learn_rates = [0.5,0.1,0.05,0.01] # initial learn_rates in case of MomentumOptimizer
l2_beta = [1e-2,1e-3,1e-4,1e-5]
dropouts = [0.5,0.4,0.2,0.1] # 1.0-keep_prob of dropout layers.
```

Finally I trained each network for 20 epochs using selected parameter set for each model on the extended data set I generated earlier using data augmentation. This gives me following set of validation accuracy for each model.  
LeNet-5 with CLAHE 0.990  
Sermanet with CLAHE 0.984  
DenseNet with CLAHE 0.997  
LeNet-5 with Mini-Net (Transnet+LeNet) 0.989  
DenseNet with Mini-Net (Transnet+DenseNet) 0.998  

So I decided to use DenseNet model with Mini-Net for colorspace transformation for my final evaluation which gives a test accuracy of 0.9960.

### Log:

2017-09-05
- MinMax Scaling [0,1]  
Standard LeNet with 3 input channels and 43 neurones at output layer.  
Validation accuracy: 0.884 at 10 EPOCHS
- Grayscale & MinMax Scaling [0,1]  
Standard LeNet with 43 neurones at output layer  
Validation accuracy: 0.902 at 10 EPOCHS

2017-09-06
- Grayscale, Histogram equalization & MinMax Scaling [0,1]  
Same network as above  
Validation accuracy: 0.918 at 10 EPOCHS

2017-09-07
- Sermanet/LeCun model. (6,16 Convolution layers with 120 hidden units in fully connected layer)  
Validation accuracy: 0.927 at 10 EPOCHS

2017-09-10
- Sermanet/LeCun model. (2LConvNet ms 22-38 + 120-feats CF classifier)  
Validation accuracy: 0.940 at 10 EPOCHS

2017-09-13
- LeNet-5 with 16 convolution filters at each stage and 256 and 64 neurons respectively at hidden fully connected layers  
BATCH_SIZE: 16  
Validation accuracy: 0.944 at 10 EPOCHS
- LeNet-5 with 16 and 40 convolution filters at stage 1 and 2 respectively and 256 and 84 hidden units respectively in fully connected layers  
Validation accuracy: 0.951 at 10 EPOCHS

2017-09-15
- Sermanet/LeCun model. (2LConvNet ms 22-38 + 120-feats CF classifier)  
BATCH_SIZE: 16  
Validation accuracy: 0.955 at 10 EPOCHS

2017-09-18
- LeNet-5 with 16 and 40 convolution filters at stage 1 and 2 respectively and 256 and 84 hidden units respectively in fully connected layers  
Dropout with keep probability 0.8 at fully connected layers  
Validation Accuracy: 0.954 at 10 EPOCHS
- L2 Regularization with beta 1e-6.  
Xavier initializer  
Validation Accuracy: 0.966 at 10 EPOCHS

2017-09-20
- Provided training set only.  
Sermanet/LeCun model. (2LConvNet ms 22-38 + 120-feats CF classifier)  
Dropout with keep probability 0.5 added for fully connected layers.  
L2 Regularization with beta=1e-6  
Xavier initializer for weights  
Validation Accuracy: 0.973 at 10 EPOCHS
 
2017-09-21  
Image preprocessing using convnets. (RGB->conv1x1x10->conv1x1x3) (Transnet+Lenet)  
- Standard LeNet with 43 neurones at output layer  
Validation accuracy: 0.896 at 10 EPOCHS
- LeNet-5 with 16 and 40 convolution filters at stage 1 and 2 respectively and 256 and 84 hidden units respectively in fully connected layers  
Validation accuracy: 0.954 at 10 EPOCHS
- Dropout with keep probability 0.8 at fully connected layers  
L2 Regularization with beta 1e-6.  
Xavier initializer  
Validation Accuracy: 0.971 at 10 EPOCHS

2017-09-23
- LeNet-5 based model with histogram equalization.  
Batch normalization with decay of 0.9.  
Validation Accuracy: 0.976 at 10 EPOCHS
- Convnet based image preprocessing (Transnet+Lenet)  
Validation Accuracy: 0.983 at 10 EPOCHS  
--  
- Sermanet/LeCun model. (2LConvNet ms 22-38 + 120-feats CF classifier)  
Validation Accuracy: 0.976 at 10 EPOCHS

2017-09-24
- Densenet model with layer size 4.  
Initial convolution feature size 16. with growth rate 12.  
Validation Accuracy: 0.992 at 10 EPOCHS
- Convnet based image preprocessing (Transnet+Densenet)  
L2 beta=0.0001  
Validation Accuracy: 0.983 at 10 EPOCHS
- Increased growth rate of densenet model to 24.  
Set kernel size of initial convolution of densenet to 5.  
Validation Accuracy: 0.993 at 10 EPOCHS
- Increased growth rate of densenet model to 24. (Transnet+Densenet)  
Set kernel size of initial convolution of densenet to 5.  
Validation Accuracy: 0.984 at 10 EPOCHS  
--  
- Transnet+Lenet model.  
Generating additional training data by flipping and random perturbations of translations,rotations and perspective transform.  
Validation Accuracy: 0.988 at 10 EPOCHS
- LeNet-5 based model with grayscale images.  
Generating additional training data by flipping and random perturbations of translations,rotations and perspective transform.  
CLAHE instead histogram equalization. (With ```tileGridSize=(4,4)```)  
Validation Accuracy: 0.985 at 10 EPOCHS  
--  
- Sermanet/LeCun model. (2LConvNet ms 22-38 + 120-feats CF classifier)  
Generating additional training data by flipping and random perturbations of rotation and perspective transform.  
CLAHE instead histogram equalization. (With ```tileGridSize=(4,4)```)  
Validation Accuracy: 0.982 at 10 EPOCHS

2017-09-25
- Transnet+Densenet model.  
Generating additional training data by flipping and random perturbations of translations,rotations and perspective transform.  
Validation Accuracy: 0.998 at 10 EPOCHS

2017-10-01
- Densenet model.  
Generating additional training data by flipping and random perturbations of translations,rotations and perspective transform.  
Validation Accuracy: 0.993 at 10 EPOCHS
- Provided training set size only with perturbations.  
```keep_prob: 0.8, perturbation_ratio:0.4 (keep_ratio:0.6), learning_rate(initial): 0.01, l2_beta:1e-4, BATCH_SIZE: 32```  
Validation Accuracy: 0.991 at 20 EPOCHS  
--  
- LeNet-5 based model.  
Provided training set size only with perturbations.  
```keep_prob: 0.8, perturbation_ratio:0.6 (keep_ratio:0.4), learning_rate: 0.001, l2_beta:1e-6, BATCH_SIZE: 16```  
Validation Accuracy: 0.981 at 20 EPOCHS
- Extended dataset  
Validation Accuracy: 0.990 at 20 EPOCHS  
--  
- Sermanet/LeCun model. (2LConvNet ms 22-38 + 120-feats CF classifier)  
Provided training set size only with perturbations.  
```keep_prob: 0.5, perturbation_ratio:0.6 (keep_ratio:0.4), learning_rate: 0.001, l2_beta:1e-6, BATCH_SIZE: 16```  
Validation Accuracy: 0.980 at 20 EPOCHS
- Extended dataset  
Validation Accuracy: 0.984 at 20 EPOCHS  
--  
- Transnet+lenet model.  
Provided training set size only with perturbations.  
```keep_prob: 0.8, perturbation_ratio:0.2 (keep_ratio:0.8), learning_rate: 0.005, l2_beta:1e-5, BATCH_SIZE: 16```  
Validation Accuracy: 0.984 at 20 EPOCHS

2017-10-02
- Transnet+densenet model.  
Provided training set size only with perturbations.  
```keep_prob: 0.8, perturbation_ratio:0.6 (keep_ratio:0.4), learning_rate(initial): 0.05, l2_beta:0.001, BATCH_SIZE: 16```  
Validation Accuracy: 0.994 at 20 EPOCHS

2017-10-05
- Densenet model.  
Extended dataset  
Validation Accuracy: 0.997 at 20 EPOCHS  
--  
- Transnet+lenet model.  
Extended dataset  
Validation Accuracy: 0.989 at 20 EPOCHS

2017-10-06
- Transnet+Densenet model.  
Extended dataset  
Validation Accuracy: 0.998 at 20 EPOCHS


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



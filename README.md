# Self-driving-car-on-Udacity-simulator
To teach a virtual car how to drive, I took the deep learning approach. Specifically, I trained a convolutional neural network (CNN) on how to drive on the simulator.
* You can download the simulator [here](https://github.com/udacity/self-driving-car-sim/tree/term3_collection).

<p align="center">
   <img  src="https://miro.medium.com/max/1400/1*2u3zy6GRNBKb5CAVNqkk9Q.png">
</p>

## The simulator had two modes:
* Training mode: In the training mode, you drive the car manually to record the driving behavior. You can use the recorded images to train your CNN.
* Autonomous Mode: In the autonomous mode, you are testing your model to see how well it can drive the car without dropping off the road / hitting obstacles. Each driving instruction contains a steering angle and an acceleration throttle, which changes the car’s direction and the speed. As this happens, the program will receive new image frames at real time.

## Capturing the data
We need to manually control the direction and speed of the car to complete the lap under the training mode. The simulator will record several information such as speed, steering angle, brake and image from the point of view of the three cameras on the car at every instance. All these information are exported as the driving log in csv file. Our driving style directly affect the data collected and this will affect our model performance.

## Extraction of data from driving log
The driving log contains path of the images and other related data. So the images from the paths in driving log are loaded and preprocessed.

## Data augmentation
* Most of the data consisted images which had steering angle in the range (-0.2,0.2). Number of steering angles in the range (-1.0,-0.2)U(0.2,1.0) data were very less in number. So to generate more data for higher steeing angle values, the images from left and right camera were used with an offset of +/-0.25 in their steering angle.
* The bigger dataset produced after adding extra images of turns was augmented by performing:
1. Random horizantal flipping- steering angle was multiplied with -1
2. Translating the image
3. Adding random shadows in the image
4. Random brightness
* An image data generator was made to augment the training data.
## The model
The most important thing that we need to predict to drive successfully is the steering angle of the vehicle. The throttle in our case can be calculated by a simple formula involving steering angle.

* The architecture of the CNN was mainly inspired from Nvidia’s [End to End Learning for Self-Driving Cars paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

<p align="center">
   <img width="200" height="360" src="https://miro.medium.com/max/1400/1*_ALA3C3qeRQgJoh3LZnFSg.png">
</p>
 ## Training
 I used a combination of [data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity and some that I collected using the training mode in the simulator. A single data example looks as follows,
 <p align="center">
   <img src="https://miro.medium.com/max/1400/1*lFZrc_-opIELSG4zEQqhSA.jpeg">
</p>

A tuple of data has 3 images left, right, centre and corresponding steering angle provided at that moment.

* We assign the angle to the central image and give the left and right images an offset when we train the network. This offset helps the car recover to the center if it veers off course.

## Performance
There were two models trained:
* First one was able to follow the road staying at the center but was not able to take very sharp turns.
* Second model was able to take sharp turns perfectly, but it was slightly wobbling on a straight track(Maybe due to too much of turning data while training it)
A weighted ensembling technique was used to get the best of both the models.
This ensembled model has lesser wobbling and was able to take sharp turns.


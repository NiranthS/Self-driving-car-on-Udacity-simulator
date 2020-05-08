# Self-driving-car-on-Udacity-simulator
* To teach a virtual car how to drive, I took the deep learning approach. Specifically, I trained a convolutional neural network (CNN) on how to drive on the simulator.
* You can download the simulator [here](https://github.com/udacity/self-driving-car-sim/tree/term3_collection).
<p align="center">
   <img  src="https://miro.medium.com/max/1400/1*2u3zy6GRNBKb5CAVNqkk9Q.png">
</p>

* The most important thing that we need to predict to drive successfully is the steering angle of the vehicle.

* The throttle in our case can be calculated by a simple formula involving steering angle.

* The architecture of the neural network was mainly inspired from Nvidia’s [End to End Learning for Self-Driving Cars paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

<p align="center">
   <img width="400" height="700" src="https://miro.medium.com/max/1400/1*_ALA3C3qeRQgJoh3LZnFSg.png">
</p>

 I used a combination of [data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity and some that I collected myself by doing training laps on the simulator. A single data example looks as follows,
 <p align="center">
   <img src="https://miro.medium.com/max/1400/1*lFZrc_-opIELSG4zEQqhSA.jpeg">
</p>

A tuple of data has 3 images left, right, centre and corresponding steering angle provided at that moment.

* We assign the angle to the central image and give the left and right images an offset when we train the network. This offset helps the car recover to the center if it veers off course.


* In simulator, negative steering angles are for left turns and positive steering angles are for right turns.

* The training data had a bias for left turns so, to remove this, all the images are flipped and the steering angle is multiplied with -1 to get an equal number of left and right turns.

* An image data generator was made to augment the training data.
* The data generator changes the brightness of the image, adds shadows, flips image sometimes, etc
* First I trained a network on the data provided by Udacity. This network was following the road properly and was able to take small turns smoothly but, it was not able to take longer/sharp turns.
* Then I made some training data in the simulator by driving the car manually which mainly consisted of sharp turns. This data was added to the previous data and a new model was trained.
* This new model was able to take turns (both sharp and small) properly but on a straight road, it was wobbling.
* A weighted ensembling technique was used to get the best of both the models.
* This ensembled model has lesser wobbling and was able to take sharp turns.


# Buildings detector
A solution for the image segmentation problem of detecting buildings on satellite images.

## The problem
The problem is semantic segmentation of satallite images by labeling each pixel as building or not.

## The data
The data is 3 channels (RGB) 650\*650 .tif images. Each image has a pair as geojson file which has coordinates of poligons for buildings. 

## Description of the problem solutioning
(all the process of work is in Buildings_detection.npynb)

As this is a segmentation problem the U-net architecture was choosen.

![U-net Neural Net Structure](https://github.com/EgShes/buildings_detector/tree/master/img/u-net.png "U-net Neural Net Structure")

The evaluation metrics was Dice coefficient 
![Dice coeff](http://latex.codecogs.com/gif.latex?%5Cfrac%7B2%20*%20%7CX%20%5Ccap%20Y%7C%7D%7B%7CX%7C%20&plus;%20%7CY%7C%7D "Dice coeff")

The loss was -Dice coefficient

(256, 256) window size was choosen. Each (650, 650) was cropped at 9 pieces.

20 % of images was hold as test set.

All the data was prerocessed with mask_tiff.py and data_processing.py before loading to learn. It allows to avoid out of memories errors.

### First architecture
First U-net was too narrow. It even could not predict. The best dice_coeff was about 0.40. Data contained too mach images without any buildings at them.

Was a lot of useless attempts because of narrowness of the net.


### Second architecture
**A** attempt

I used an U-net implamentation of [ZFTutbo](https://github.com/ZFTurbo). 

The dice_coeff will be replaced with its customization ![Custom dice coeff](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%7CX%20%5Ccap%20Y%7C%7D%7B%7CX%7C%20&plus;%20%7CY%7C%20-%20%7CX%20%5Ccap%20Y%7C%7D 'Custom dice coeff')

But as I knew after training the absance of images with no buildings at them is bad. Net was always making mistakes on them (forest, crops).

During training the best dice coeff was about 0.69. On test set it was 0.33.

**B** attempt

For this attempt train and val data contained about 15% of images with no houses.

During training best val dice coeff was less under 0.70

The result was really good. On test set dice coeff was 0.62. 

All the result masks are in results folder.

## Experiments and Weights

The weights are orginized as the results of experiments.

### 1 Experiment (first architecture)

* **a** 30 epoch, dice 0.22
* **b** 25 epoch, dice 0.266
* **c** 25 epoch, minor changes in net (found typo mistakes in number of features)

### 2 Experiment (first architecture)

* **a** 0 nobuilding images, 15 epochs, batch=32, dice = 0.30
* **b** 0 nobuilding images, 30 epochs, batch=16, lr = 0.001, added LRReduceOnPlatau dice = same

### 3 Experiment (second architecture)

* **a** 0 nobuilding, net is good, dice = 0.68, mistakes on forests and crops (nobuildings areas) test dice = 0.33
* **b** 15% of nobuildings, success, dice = 0.70, test dice = 0.63, result test masks in result folder

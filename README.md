# Melanoma-Classification-using-transfer-learning

Malignant melanoma is a form of skin cancer that develops from melanocytic cells in the human body and can prove to be dangerous if not treated early. With the advent of artificial intelligence, deep learning approaches are being applied for the diagnosis of the disease which in turn shall help medical professionals in the line of treatment.  In this paper, the concept of transfer learning is used by implementing a pre-trained VGG19Net (pre-trained with ImageNet dataset) and further using certain layers of the pre-trained model for analysis on the dataset of our interest by freezing rest of the convolutional layers. Here, images from the ISBI2016 challenge dataset are taken and used for classification which records a commendable validation accuracy of 81.33% along with a testing accuracy of 86.67%, precision of 95.08%, recall of 82.25%, IoU of 78.26% and Dice-score of 85.29%.

# Experimental Procedure

![VGG19](https://user-images.githubusercontent.com/66628385/89116017-a1120e80-d4ac-11ea-881c-830625a73062.PNG)

Fig. The custom CNN, also known as VGG19 model is used for the purpose of training and testing.

## Procedure used:
### 1. Data Pre-Processing:
Here, data from ISBI 2016 is taken and sampled first to a sample size as follows:
346 images from class 0 and 173 images from class 1(later oversampled to 346 to balance the dataset) for training set.
For validation set, first 75 images from the validation set of each class are taken.
For testing set, first 75 images from each class of ISBI 2017 dataset are taken.
Data images are augmented by horizontal rotation and increase in saturation.

### 2. Transfer Learning of CNN:
Here, a pre-trained VGG19 network (pre-trained with ImageNet dataset) is taken and trained with the new dataset and validated. Once results are stored and observed, classification begins.

### 3. Classification:
Here, the pre-trained VGG19 network is tested against the 150 image dataset extracted from ISBI 2017 dataset and lassified accordingly. The results of the performance matrix are stored and classification results are displayed.

## Inventory:
a. VGG19 network code

b. classification code

c. image resize code

d. results

e. readme

f. license

## Keywords:
Melanoma, Transfer Learning, Deep Learning, Augmentation, VGG19.


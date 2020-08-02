# Melanoma-Classification-using-transfer-learning

Malignant melanoma is a form of skin cancer that develops from melanocytic cells in the human body and can prove to be dangerous if not treated early. With the advent of artificial intelligence, deep learning approaches are being applied for the diagnosis of the disease which in turn shall help medical professionals in the line of treatment.  In this paper, the concept of transfer learning is used by implementing a pre-trained VGG19Net (pre-trained with ImageNet dataset) and further using certain layers of the pre-trained model for analysis on the dataset of our interest by freezing rest of the convolutional layers. Here, images from the ISBI2016 challenge dataset are taken and used for classification which records a commendable validation accuracy of 81.33% along with a testing accuracy of 86.67%, precision of 95.08%, recall of 82.25%, IoU of 78.26% and Dice-score of 85.29%.

# Experimental Procedure

![VGG19](https://user-images.githubusercontent.com/66628385/89116017-a1120e80-d4ac-11ea-881c-830625a73062.PNG)

Fig. The custom CNN, also known as VGG19 model is used for the purpose of training and testing.

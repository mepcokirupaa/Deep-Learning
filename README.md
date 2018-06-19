# Deep-Learning
Biomarkers for the identification of acute leukaemia
Abstract
----------
 In the era of digital microscopic imaging, the image processing, data analysis, classification and decision support systems have emerged as one of the most important tools for diagnostic research. Physicians can observe cellular internal structures and their abnormalities by visualizing and analyzing images. Leukemia is a malignant disease characterized by the uncontrolled accumulation of abnormal white blood cells. The recognition of acute leukemia blast cells in colored microscopic images is a challenging task. Identification and diagnosis of these types of abnormalities by humans is difficult and may lead to misidentification. Therefore an automatic system for the identification would be of great help. The first important step in the automatic recognition of the acute leukemia is image segmentation, which is considered to be the most critical step. Here, a clinical relevance setup is designed that includes the panel selection, segmentation using K - Means Clustering to identify the leukemia cells, feature extraction and classification using Convolutional Neural Network. After the decision support system successfully identifies the cells and its internal structure, the cells are classified according to their morphological features. This system was tested using a public dataset for Acute Lymphoblastic Leukemia. The algorithm was applied to the dataset of 108 images. The testing of this system using this dataset demonstrated an overall accuracy of 99.5%.  
 
 Dataset Description
 -------------------------
Name                   :  ALL_IDB1
Total Size             :  144 MB
Total No. of Images    :  108
Size of one Image      :  528 KB 
Format                 :  JPG
Colour Depth           :  24 bit 
Resolution             :  1592 x 944
Unhealthy Individuals  :  75
Healthy Individuals    :  33


Steps
---------
   PreProcessing steps are done through Matlab.
   After Pre-Processing input files given to Python for Deep Learning.

MODULES
----------
  1. Segmentation

  2. Object Enhancement
	
  3. Feature Extraction and Classification
  
Segmentation
----------------------
  The images are represented by the three colour components, RGB i.e., the Red, Green and Blue value representation. The histogram of the green colour distribution shows that the green channel contains the most contrast information about the target cells. This plane is selected for the subsequent segmentation step. The segmentation is done based on the K-Means algorithm where K is the number of pre-selected groups. It begins by partitioning a vast arrangement of vectors into groups having the same number of points. The centroid point represents the group. 

Here, the K-means Clustering Algorithm is applied to separate the desired cells successfully through two steps as follows: 
1.	Dividing green intensity components of the image into three classes as shown in Fig 4.1, each class represents a component of the image extracted by applying the K-means algorithms with K=3.
The three clusters extracted are
•	Background (Fluid)
•	Non - target cells (Platelets)
•	White blood cells (Target cells)

To retrieve the cluster with the target cells, the cluster with the minimum centre value is found and passed to the next step.

2.	The former component retrieved from the previous step is re – divided into two classes with  K = 2 as shown in Fig 4.1.
The two clusters extracted are
•	Cells 
•	Nucleus

OBJECT ENHANCEMENT
-----------------------
	Unwanted regions are present in the segmented image . Image problems, such as cell overlapping and cell distortion should be solved. Identifying the overlapped objects as a single object leads to errors in measurements and statistics. 
The Watershed Transform is used to mark the foreground objects and background locations. It is used here to separate the overlapped objects like cells .

For each pixel, a value is assigned. The value assigned is the distance between the respective pixel and the nearest non zero pixel. Then, the Euclidean Distance Transform is used for distance calculation. The watershed transform finds watershed ridge lines in an image by treating it as a surface where light pixels are considered as high elevation and dark pixels as low elevation. The values returned are integer values 0 or greater than 0. If it is 0, that region does not belong to a unique watershed region. If it is 1, that region belongs to first watershed region. If it is 2, that region belongs to second watershed region and so on.
The following Morphological Operations are applied to get the correct shape.
•	Erosion and Dilation Operations.
•	Closing and Opening Operations.

Feature Extraction and Classification
----------------------------------------
    The feature extraction facilitates both the classification and recognition of leukemia cells. The technique of Convolutional Neural Network (CNN) is used to extract the features and classify the output. 

The CNN eliminates the need for manual feature extraction. So there is no need for the identification of features manually to classify the images. This automated feature extraction makes deep learning models highly accurate for computer vision tasks.  

Data Augmentation

	Deep Learning needs large amount of images. If dataset contains only small number of images, the network may over fit. To prevent over fitting data augmentation is done. During the training of the datasets, the following augmentation steps are done.
•	The images are rotated into 20 degrees.
•	The images are randomly translated vertically or horizontally using width-shift-range and height-shift-range with a range of 0.2.
•	Random shearing transformation with range 0.2.
•	Random flipping of half of the images horizontally.
•	Random flipping of half of the images vertically.
	16 batches were generated for each image. So approximately 1600 images were created from the 108 existing images.

Convolutional Layer
---------------------
 Takes original data and create feature maps from it.
 Convolution preserves the spatial relationship between pixels by learning image features using filters. 
 Filters are feature detectors.
 Different filters can detect different features.
   E.g., edges, cell colour, texture.
   Three convolutional layers have been used:
        The first layer has 3 x 3 x 32 filters.
        The second layer has 3 x 3 x 32 filters.
        The third layer has 3 x 3 x 32 filters.
Pooling
----------
  Pooling reduces the dimensionality of each feature map but retains the most important information. 

        E.g. Max, Avg, Sum.
   Most common form of pooling is Max Pooling.
   Three max - pooling layers each of size 2 x 2 have been used.
   
Fully Connected Layer
--------------------------
   The fully connected layer is a traditional Multilayer Perceptron that uses a sigmoid activation function in the output layer. The term fully connected implies that every neuron in the previous layer is connected to every neuron on the next layer.

The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the fully connected layer is to use these features for classifying the input image into various classes based on the training dataset. The output of the last convolutional layer is converted into 1D vector. The fully connected layer can accept only the 1D vector. 
Apart from classification, adding a fully-connected layer is also a cheap way of learning non-linear combinations of these features.      
 CONCLUSION
 -----------------
   This application develops a technique for automatically identifying a person with acute leukemia. The mistakes that can occur during manual analysis are reduced. The overall time taken to evaluate and analyze the cells and their behavior is minimized. This system produces an overall accuracy of 99.5%.
This project can be further extended to develop a robust segmentation system to identify the different sub types of leukemia.



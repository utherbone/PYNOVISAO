## __Title: Pynovisao__
#### Authors (email):
- Adair da Silva Oliveira Junior
- Alessandro dos Santos Ferreira
- Diego André Sant'Ana (diegoandresantana@gmail.com)
- Diogo Nunes Gonçalves (dnunesgoncalves@gmail.com)
- Everton Castelão Tetila (evertontetila@gmail.com)
- Fabio Prestes Cesar Rezende (fpcrezende@gmail.com)
- Felipe Silveira (eng.fe.silveira@gmail.com)
- Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
- Gilberto Astolfi (gilbertoastolfi@gmail.com)
- Hemerson Pistori (pistori@ucdb.br)
- Joao Vitor de Andrade Porto (jvaporto@gmail.com)
- Nícolas Alessandro de Souza Belete (nicolas.belete@gmail.com)

## Resume:

Computer Vision Tool Collection for Inovisão. This collection of tools allows the user to select an image (or folder) and realize numerous actions such as:
- Generate new Datasets and classes
- Segmentation of images
- Extract features from an image
- Extract frames from videos
- Train Machine Learning algorithms
- Classify using CNNs
- Experiment with data using Keras
- Create XML files from segments previously created.

## Open Software License: 

NPOSL-30 https://opensource.org/licenses/NPOSL-3.0 - Free for non-profit use (E.g.: Education, scientific research, etc.). Contact Inovisão's Prof. Hemerson Pistori (pistori@ucdb.br), should any interest in commercial exploration of this software arise.

## How to cite:

[1] dos Santos Ferreira, A., Freitas, D. M., da Silva, G. G., Pistori, H., & Folhes, M. T. (2017). Weed detection in soybean crops using ConvNets. Computers and Electronics in Agriculture, 143, 314-324.

## How to Install

- **Option 1, Linux-only Script**

You can easily install Pynovisão utilizing the automated installation script given with it, as seen by the following steps:

- From inside of this directory:
```
 [...]/pynovisao
```

- Execute the following command:
```
$ sudo bash INSTALL.sh
```
**NOTE**: This script has been tested for Ubuntu versions 19.04 and 18.04

- **Option 2, without INSTALL.sh**

Besides it's dependencies, Python 2.7.6 or Python 3.6 is needed. (Latest tested versions for this software)

- Installing the necessary dependencies on Python 3.6:
```
$ sudo apt-get update
$ sudo apt-get install libfreetype6-dev tk tk-dev python3-pip openjdk-8-jre openjdk-8-jdk weka weka-doc python3-tk python3-matplotlib
$ source ~/.bashrc
$ sudo pip3 install numpy
$ sudo pip3 install -r requirements_pip3.txt
$ sudo pip3 install tensorflow 
$ sudo pip3 install keras
```

- Installing the necessary dependencies on Python 2.7:
```
$ sudo apt-get update
$ sudo apt-get install libfreetype6-dev tk tk-dev python-pip openjdk-8-jre openjdk-8-jdk weka weka-doc python-tk python-matplotlib
$ source ~/.bashrc
$ sudo pip install numpy
$ sudo pip install -r requirements_pip3.txt
$ sudo pip install tensorflow 
$ sudo pip install keras
```

## How to install Caffe ( Optional )

#### Ubuntu / Windows
In order to use the CNNCaffe classifier, a ConvNet based on the AlexNet topology, it is necessary to install Caffe.

It's installation is more complex than the ones previously mentioned, and more detailed instructions can be found below:
-  http://caffe.berkeleyvision.org/installation.html

After installing Caffe, in order to realize classification with it you will need to train it with Pynovisão using the command line, since there currently is no interface for ConvNet Training.

The tutorial for training can be found below:
- http://caffe.berkeleyvision.org/gathered/examples/imagenet.html

Finally, it is necessary to configure your CNNCaffe.
- For the fields *ModelDef, ModelWeights* and *MeanImage*, you must supply the relative paths to the traning done previously.
- For the field *LabelsFile* you must supply the path to a file that describes all the classes in order (0, 1, 2, ..., n-1; where n is the number of classes trained).
- A example file can be found in **[...]/pynovisao/examples/labels.txt**.

## How to use:

#### Opening the software
- In order to download Pynovisao, click the download button in the top right of the screen (Compressed folder), or type the following command in a terminal:
```
 $ git clone http://git.inovisao.ucdb.br/inovisao/pynovisao
```

- From inside of this directory:
```
 [...]/pynovisao
```

- Enter the folder named **[...]/pynovisao/src** or type the following command in the terminal to do so:
```
 $ cd src
```

- Next, type the following command if you desire to run it using Python 2.7:
```
 $ python main.py
```
- Or, should you want to run it using Python 3.6:
```
 $ python3 main.py
```

Now you are able to run Pynovisão!
    
#### Other options:

- Show All options available

```
 $ python main.py --help
```

- Executes the program, defining the wanted classes and it's respective colours (X11 colour names)

```
 $ python main.py --classes "Soil Soy Grass LargeLeaves" --colors "Orange SpringGreen RebeccaPurple Snow"
```

- A Linux script exists in *[...]/pynovisao/src/util* to help divide images into training, validation and testing datasets. It has not been implemented to the main GUI. In order to use it, use the folowwing commands:

```
 $ cd src/util
 $ chmod 755 split_data.sh
 $ ./split_data -h
```
# Features of this software:

## File
### Open Image (Shortcut: Ctrl + O)
Opens a file selection windows and allows the user to choose a desired image to work upon.
### Restore Image (Shortcut: Ctrl + R)
Restores the selected image to it's original state.
### Close Image (Shortcut: Ctrl + W)
Closes the currently selected image.
### Quit (Shortcut: Ctrl + Q)
Closes Pynovisão.

## View
### Show Image Axis (Shortcut: Not Defined)
Shows a X/Y axis on the Image.
### Show Image Toolbar (Shortcut: Not Defined)
Shows a list of all the images in the selected folder.
### Show Log (Shortcut: Not Defined)
Shows a log with information about the current processes and Traceback errors should they happen.

## Dataset
### Add new class (Shortcut: Ctrl + A)
Create a new class. This will create a new folder in the /data folder.
### Set Dataset Path (Shortcut: Ctrl + D)
Choose the folder with the desired images.
### Dataset Generator (Shortcut: Not Defined)
Creates a new dataset utilizing the selected folder.

## Segmentation
### Choose Segmenter (Shortcut: Not Defined)
Choose the desired segmentation method. Please research the desired method before segmenting. The Default option is SLIC.
### Configure (Shortcut: Ctrl + G)
Configure the parameters for the segmentation.
- Segments: Number of total segments the image should be split into.
- Sigma: How "square" the segment is.
- Compactness: How spread out across the image one segment will be. A higher compactness will result in more clearly separated borders.
- Border Color: The color of the created segments' borders. This is only visual, it will not affect the resulting segment.
- Border Outline: Will create a border for the segment borders.
### Execute (Shortcut: Ctrl + S)
Execute the chosen segmentation method with the desired parameters.
Once Segmented, the user can manually click on the desired segments and they will be saved in data/demo/**name-of-the-class**/**name-of-the-image**_**number-of-the-segment**.tif.
### Assign using labeled image (Shortcut: Ctrl + L)
Use a mask/bicolor image created using a labelling software (LabelMe/LabelImg) and applies it to the original/selected image, and generates all the correct segments inside such mask.
### Execute folder (Shortcut: Not Defined)
Same as the Execute command, however it realizes the segmentation on an entire folder at once.
### Create .XML File (Shortcut: Not Defined)
Will create a .xml file using the chosen segments. The .xml will be saved in data/XML/**name-of-the-image**.xml

## Feature Extraction
### Select Extractors (Shortcut: Ctrl + E)
Select the desired extractors to use. The currently available extractors are:
- Color Statistics;
- Gray-Level Co-Ocurrence Matrix;
- Histogram of Oriented Gradients;
- Hu Image Moments;
- Image Moments (Raw/Central);
- Local Binary Patterns;
- Gabor Filter Bank;
- K-Curvature Angles.
Please research what each extractor does, and choose accordingly. By default all extractors are chosen.
### Execute (Shortcut: Ctrl + F)
Execute the chosen Extractors. It will create a training.arff file in the data/demo folder.
### Extract Frames (Shortcut: Ctrl + V)
Will extract frames from a video. The user must choose the folder where the desired videos are, and the destination folder where the consequent frames will be extracted to.

## Training
### Choose Classifier (Shortcut: Not Defined)
Choose the desired classifier to use. Only one can be chosen at a time.
- CNNKeras
- CNNPseudoLabel
- SEGNETKeras
If the user is interested in implementing it's own classifiers into Pynovisão, please go to **Implementing a new classifier in Pynovisão**
### Configure (Shortcut: Not Defined)
Choose the desired parameters for the currently selected classifier.
Each classifier has it's own parameters and configurations, and therefore must be extensibly research should the desired result be achieved.
### Execute (Shortcut: Ctrl + T)
Train the selected classifier utilizing al the chosen parameters and the training.arff file created previously.

## Classification
### Load h5 weights (Shortcut: Not Defined)
*Only used for CNN classifiers* Take a previously created weight .h5 file and use it for this classification.
### Execute (Shortcut: Ctrl + C)
Execute the current classifier over the currently selected image.
### Execute folder (Shortcut: Not Defined)
Same as the previous command, however executes all the image files inside a selected folder at once.

## Experimenter
### Ground Truth (Shortcut: Not Defined)
Utilizes the currently selected image as the ground truth for the experimentations.
### Execute Graphical Confusion Matrix (Shortcut: Not Defined)
For each classifier, creates a graphic with it's confusion matrix for the choen dataset.
### Cross Validation (Shortcut: Ctrl + X)
Performs cross validation utilizing the previously experimented classifiers.
### Experimenter All (Shortcut: Ctrl + P)
Runs all Weka classifiers and experiments with them.

## XML
### Configure folders (Shortcut: Not Defined)
Choose the target folder for the original images and the other target folder for the segments to be searched and conevrted into a .xml file.
### Execute Conversion (Shortcut: Not Defined)
Executes the conversion using the two given folders. The file with the annotations will be saved in *[...]/pynovisao/data/XML*, with the name ***image** + .xml*. 

# Implementing a new classifier in Pynovisão

In this section we shall show the steps needed to implement a new classifier into Pynovisão. As an example, we are using **Syntactic**, of type **KTESTABLE** and vocabulary size as an hyperparameter.

Inicially, you need to create a class where all the types of your classifier are in a dictionary (Key, Value). The class must be created inside *[...]/pynovisao/src/classification/*. As an example, look for the *SyntacticAlias* in *[...]/pynovisao/src/classification/syntactic_alias.py*.

The next step is creating the .py file for your classifier in your directory *[...]/pynovisao/src/classification/*, for example, *syntactic.py*.
In this newly-created file you must implement your classifier class extending the class **Classifier**, which is implemented in the file *[...]/pynovisao/src/classification/classifier.py*.
See the example below:

```python
#syntactic.py
#minimal required imports
from collections import OrderedDict
from util.config import Config
from util.utils import TimeUtils
from classifier import Classifier

class Syntactic(Classifier):
    """Class for syntactic classifier"""
```

In the contructor class you must inform default values for the parameters. In the case fo the example below, **classname** is the type of classifier and **options** is the size of the alphabet. Besides, some attributes must be inicialized: **self.classname** and **self.options**. The attribute **self.dataset** (optional) is the path to the training and testing dataset which tells the user in the GUI. Having this attribute in the class is important to get access to the dataset in any of the methods and is initialized in the method **train** discussed later.

```python
def __init__(self, classname="KTESTABLE", options='32'):

        self.classname = Config("ClassName", classname, str)
        self.options = Config("Options", options, str)
        self.dataset = None
        self.reset()
```

The methods **get_name**, **get_config**, **set_config**, **get_summary_config** and **must_train** have default implementations, as seen in example in *[..]/pynovisao/src/classification/classifier.py*.

The **train** method must be implemented in order to train your classifier. The **dataset** parameter is given the path to the training images. Within the method, the value of the attribute self.dataset, declared as optional in the constructor, is altered to the current training directory.

```python
def train(self, dataset, training_data, force = False):              
        
        dataset += '/'
        # Attribute which retains the dataset path.
        self.dataset = dataset 
  	    # The two tests below are default.
        
        if self.data is not None and not force:
            return 
        
        if self.data is not None:
            self.reset()
		
	   # Implement here your training.
```

The **classify** method must be implemented should you want your classifier to be able to predict classes for images. The **dataset** parameter is given the training images, and **test_dir** is given the temporary folder path created by Pynovisão, where the testing images are located. This folder is created within the **dataset** directory and, to acesss it, just concatenate **dataset** and **test_dir** as show in the example below. The parameter test_data is a .arff file with data for the testing images.
 
 This method must return a list containing all the predicted classes by the classifier. E.g.: [‘weed’,’weed’,’target_stain’, ‘weed’]

```python
def classify(self, dataset, test_dir, test_data):
      
	   # Directory retaining the testing images.
       path_test = dataset + '/' + test_dir + '/'        
        
       # Implement heere the prediction algorithm for your classifier.
 
       return # A list with the predicted classes
```

The **cross_validate** must be implemented and return a string (info) with the metrics.
Obs.: The attribute **self.dataset**, updated in **train**, can be used in **cross_validate** to access the training images folder.

```python
def cross_validate(self, detail = True):
        start_time = TimeUtils.get_time()        
        info =  "Scheme:\t%s %s\n" % (str(self.classifier.classname) , "".join([str(option) for option in self.classifier.options]))
	  
	   # Implement here the cross validation.
	   return info
```
The **reset** method must also be implemented in default form, as seen below.

```python
def reset(self):
        self.data = None
        self.classifier = None
```

After implementing your classifier, you must configure it in Pynovisão by modifying **[...]/pynovisao/src/classification/__init__.py**.

Should utility classes be necessary, they must be created in **[...]/pynovisao/src/util/**. They must also be registered as modules in **[...]/pynovisao/src/util/__init__.py**.


Should any problem related to the number of processes arise, add these two variables in your terminal:

```
export OMP_NUM_THREADS=**number of threads your cpu has**
export KMP_AFFINITY="verbose,explicit,proclist=[0,3,5,9,12,15,18,21],granularity=core"
```


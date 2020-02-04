[//]: # (Image References)

[image1]: dog_pred_example.png "Sample Output"

# Dog Breed Classifier

In this project, I build a neural network pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If an image of a human face is supplied, the code will identify the resembling dog breed. The human and dog identification algorithm is based on Convolutional Neural Networks (CNN).

In this real-world setting, a series of models must fit together to perform different tasks; for instance, the algorithm that detects a human face in an image will be different from the CNN that infers a dog breed.

Human face detection is realized via an [cv2 face detector](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) as well as a “Multi-Task Cascaded Convolutional Neural Network,” or MTCNN for short, described by [Kaipeng Zhang, et al.](http://kpzhang93.github.io/) in the 2016 paper titled “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks”.

The dog breed classification is realized in two different approaches:
1. by using an own CNN based architecture as a model with a layer combination of three times 'Conv-ReLU-MaxPool', deeply enough for a feature extraction and an appropriate image size/feature reduction. However, the accuracy is limited to 15%.
2. by using a CNN based pretrained VGG19 model from Torchvision via Transfer Learning. VGG19 enables a classification of the 133 possible dog breeds in the dataset. Only the last layer (the classifier) of VGG19 is exchanged to adjust the number of classes. The number of classes in the VGG19 approach is 1000. In the dog-breed-project there are only 133 classes. Hence, only the number of 'out_features' in classifier(6) is replaced by 133.

![Sample Output][image1]

This is a project of the Udacity Nanodegree program 'Deep Learning'. Please check this [link](https://www.udacity.com/course/deep-learning-nanodegree--nd101) for more information.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- To run this script you will need to use a Terminal (Mac OS) or Command Line Interface (Git Bash on Windows).
- If you are unfamiliar with Command Line check the free [Shell Workshop](https://www.udacity.com/course/shell-workshop--ud206) lesson at Udacity.


### Installing (via pip)

- To run this script you will need to use a Terminal (Mac OS) or Command Line Interface (e.g. Git Bash on Windows [git](https://git-scm.com/)), if you don't have it already.
- If you are unfamiliar with Command Line coding check the free [Shell Workshop](https://www.udacity.com/course/shell-workshop--ud206) lesson at Udacity.

1. Clone this repository by opening a terminal and typing the following commands:

```
$ cd $HOME  # or any other development directory you prefer
$ git clone https://github.com/ddhartma/Dog-Breed-Classifier.git
$ cd project-tv-script-generation_cloned_2
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.
4. Make sure you have already installed the necessary Python packages according to the README in the [program repository](https://github.com/udacity/deep-learning-v2-pytorch.git).


Of course, you obviously need Python. Python 3 is already preinstalled on many systems nowadays. You can check which version you have by typing the following command (you may need to replace python3 with python):

```
$ python3 --version  # for Python 3
```
Any Python 3 version should be fine, preferably 3.5 or above. If you don't have Python 3, you can just download it from [python.org](https://www.python.org/downloads/).

You need to install several scientific Python libraries that are necessary for this project, in particular NumPy, Matplotlib, Pandas, Jupyter Notebook, Torch and Torchvision. For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew). The advantage of using pip is that it is easy to create multiple isolated Python environments with different libraries and different library versions (e.g. one environment for each project). The advantage of using your system's packaging system is that there is less risk of having conflicts between your Python libraries and your system's other packages. Since I have many projects with different library requirements, I prefer to use pip with isolated environments.

These are the commands you need to type in a terminal if you want to use pip to install the required libraries. Note: in all the following commands, if you chose to use Python 2 rather than Python 3, you must replace pip3 with pip, and python3 with python.

First you need to make sure you have the latest version of pip installed:

```
$ python3 -m pip install --user --upgrade pip
```
The ```--user``` option will install the latest version of pip only for the current user. If you prefer to install it system wide (i.e. for all users), you must have administrator rights (e.g. use sudo python3 instead of python3 on Linux), and you should remove the ```--user``` option. The same is true of the command below that uses the ```--user``` option.

Next, you can optionally create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially very different libraries, and different versions:

```
$ python3 -m pip install --user --upgrade virtualenv
$ python3 -m virtualenv -p `which python3` env
```

This creates a new directory called env in the current directory, containing an isolated Python environment based on Python 3. If you installed multiple versions of Python 3 on your system, you can replace ```which python3``` with the path to the Python executable you prefer to use.

Now you must activate this environment. You will need to run this command every time you want to use this environment.

```
$ source ./env/bin/activate
```

On Windows, the command is slightly different:

```
$ .\env\Scripts\activate
```

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the --user option (alternatively you could install the libraries system-wide, but this will probably require administrator rights, e.g. using sudo pip3 instead of pip3 on Linux).

```
$ python3 -m pip install --upgrade -r requirements.txt
```

Great! You're all set, you just need to start Jupyter now.

## Running the tests

The following files were used for this project:

- dog_app.ipynb
- workspace_utils.py
- dog images: path/to/dog-project/dogImages
- human faces: path/to/dog-project/lfw

Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
```
jupyter notebook dog_app.ipynb
```

__NOTE:__ In the notebook, you will need to train CNNs in PyTorch.  If your CNN is taking too long to train, feel free to pursue one of the options under the section __Accelerating the Training Process__ below.



## (Optionally) Accelerating the Training Process

If your code is taking too long to run, you will need to either reduce the complexity of your chosen CNN architecture or switch to running your code on a GPU.  If you'd like to use a GPU, you can spin up an instance of your own:

#### Amazon Web Services

You can use Amazon Web Services to launch an [EC2 GPU instance](https://aws.amazon.com/de/ec2/). However, this service is not for free.

## Acknowledgments
* This is a project of the Udacity Nanodegree program 'Deep Learning'. Please check this [link](https://www.udacity.com/course/deep-learning-nanodegree--nd101) for more information.

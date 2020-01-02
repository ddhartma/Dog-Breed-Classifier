[//]: # (Image References)

[image1]: ./images/sample_dog_output2.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


# Dog Breed Classifier

In this project, I build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- To run this script you will need to use a Terminal (Mac OS) or Command Line Interface (Git Bash on Windows).
- If you are unfamiliar with Command Line check the free [Shell Workshop](https://www.udacity.com/course/shell-workshop--ud206) lesson at Udacity.


### Installing

#### Cloning to and preparing the local repository

1. Clone the repository and navigate to the downloaded folder.

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.
4. Make sure you have already installed the necessary Python packages according to the README in the [program repository](https://github.com/udacity/deep-learning-v2-pytorch.git).


#### Install jupyter notebook
First update pip via
```
pip3 install --upgrade pip
```
Then install the Jupyter Notebook using:
```
pip3 install jupyter
```
(Use pip if using legacy Python 2.)

For further help installing Jupyter Notebook check [Jupyter-ReadTheDocs](https://jupyter.readthedocs.io/en/latest/install.html)


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

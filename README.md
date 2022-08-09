# IRP

<p align="center">
  <img src="https://user-images.githubusercontent.com/90031508/183531098-494a5819-7714-4f72-8ff8-d038982eb5f0.png" alt="Water Oracle logo"/>
</p>

## WaterNets: Multi-view Learning for classifying water bodies through combining the Landsat-8, Sentinel-1 and digital elevation models bands experimentally

A Repository for independent research project

### Introduction

Water bodies classification has often been achieved through remote sensing using either the Landsat-8 or the sentinel-1 using the threshold or deep learning methods and each having different benefits. A few studies have examined how different combinations of the features will give an improved prediction, but to date no studies have used the multiview learning U-Net to explore the different combinations between the features. 

In this project, a total of 172 experiments were carried out using google earth engine and google cloud platform. Consequently, the proposed Multi-view learning neural network, entitled the ThWaterNet and GlobalWaterNet (combined Landsat-8 with Sentinel-1 data and slope) give an outstanding prediction accuracy in Thailand and globally both reaching F1 scores of around 0.97. In addition, through multiple experiments it was discovered that it is statistically significant that features paired through multiple views will give a better performance than stacking the features. It was also discovered that adding the terrain and elevation from NASADEM can improve prediction. The landsat-8 gives better predictions than sentinel-1, but when sentinel-1 is stacked with the NASADEM data, the performance can improve. In particular, if bands from sentinel-1 are stacked with slope the global average F1 score can improve from 0.61 to 0.93. Furthermore, it was discovered that even though the elevation features paired with landsat-8 through feature stacking gives a poor predicting, pairing through multiple views can increase the global F1 score to 0.96. 

This work can serve as a useful tool for government to gain a greater understanding of the hydrological system and is a powerful tool to map the extent of flood in order to understand the damage and also can be used to help predict the risk of flooding by property insurance companies.

## Important Note

⚠️ The main purpose of this repository is to train and run 172 variations of U-Net models, for a better viewing experience, the exported GEE assets have been linked to a web application in react and node js. This can be found in https://geeimperial.herokuapp.com/ or accessing the public https://github.com/edsml-kl121/geeimperial repository.

### How to run the notebooks
To use the notebooks please install all the prequisites

### Prerequisites
To run the notebooks the following must be installed:

- Google account and logins
- Google colab subscription with pro or pro+ is optional but would help with long runtime
- Google cloud platform account in order to use google cloud bucket. (Note that you would need sufficient funds to store large amount of models and training data.)
- Wandb.ai account which is free of charge


### Installation guide
Firstly, clone the repository:
```
git clone git@github.com:ese-msc-2021/irp-kl121.git
```
and `cd` into irp-kl121

Now create and activate the conda environment
```
conda env create -f environment.yml
conda activate wateroracle
```

Install required packages
```
pip install -r requirements.txt
```

Activate the setup.py in order to create tools module with
```
pip install -e .
```

Due to the tensorflow installation issue on mac M1 chip and the difficulty in communication with google cloud bucket due to the need to host virtual machines and having permission to authenticate earth engine automatically. The tests are splitted into automatic testing and manual testing. 

Automatic testing are tests on mainly `config.py` and `metrics_.py`. Manual testings are on other python files such as `losses_.py`, `model.py`, `preprocessing.py` and `sampling.py`.

For automated github pytest workflow we can do
```
pytest tools/tests/
```

For manual pytest please open the pytest.ipynb in google colab, but you would need to install <b>prequisites</b> first.

### User instructions
#### How to run the notebooks:
There are in total of 11 main notebooks serving different functions and in order to run these notebooks you must install the prequisites first

The notebook 1,2,3 and 4,5,6 should be ran in chronological order. Notebook 7 and 8 is used to tune the best models. Notebook 10 is used visualize the
performance of all the models and performing analysis. To visualize the prediction use notebook 9 and to test the custom tool package use notebook 11

1. <b>Preprocessing_and_export.ipynb</b> - Exporting training data locally in Thailand to google cloud bucket by useingearth engine's package to export multiple training patches that will be used to train models in the other notebook (TrainModels.ipynb)
2. <b>TrainingModels.ipynb</b> - The purpose of this notebook is to train the models that has been stored in the google cloud buckets by the `Preprocessing_and_export.ipynb` notebook. The training is accompanied with Wandb and is splitted into three different methods of training featurestack, multiview learning with 2 and 3 inputs variation of UNET.
3. <b>Test_accuracy_assessment.ipynb</b> - The purpose of this notebook is to evaluate the performance of our trained neural network on different metrics including recall, precision, F1 score and accuracy. The models have been created in `TrainModels.ipynb` and stored in the google cloud bucket. The metrics evaluation is accompanied with Wandb. Here we will evaluate 4 different performances NDWI on the tests data in 10 different locations, Feature stack on the tests data in 10 different locations, Multiview learning with 2 inputs on the tests data in 10 different locations and Multiview learning with 3 inputs on the tests data in 10 different locations


4. <b>Preprocessing_and_export_global.ipynb</b> - Exporting training data globally to google cloud bucket by using earth engine's package to export multiple training patches that will be used to train models in the other notebook (TrainModels_global.ipynb)
5. <b>TrainingModels_global.ipynb</b> - The purpose of this notebook is to train the models that has been stored in the google cloud buckets by the `Preprocessing_and_export_global.ipynb` notebook. The training is accompanied with Wandb and is splitted into three different methods of training featurestack, multiview learning with 2 and 3 inputs variation of UNET.
6. <b>Test_accuracy_assessment_global.ipynb</b> - The purpose of this notebook is to evaluate the performance of our trained neural network on different metrics including recall, precision, F1 score and accuracy. The models have been created in `TrainModels_global.ipynb` and stored in the google cloud bucket. The metrics evaluation is accompanied with Wandb. Here we will evaluate 3 different performances Feature stack on the tests data in 10 different locations, Multiview learning with 2 inputs on the tests data in 10 different locations and Multiview learning with 3 inputs on the tests data in 10 different locations

7. <b>Hyperparameter_tuning.ipynb</b> - The main purpose of this notebook is to do hypermeter tuning on the loss functions and the dropout probability in the variation of the UNET to obtain the best prediction in Thailand. 

8. <b>Hyperparameter_tuning_global.ipynb</b> - The main purpose of this notebook is to do hypermeter tuning on the loss functions and the dropout probability in the variation of the UNET to obtain the best prediction in globally. 

9. <b>ImageExport.ipynb</b> - The main purpose of this notebook is to export image in regions of interest and predict this image and generate the output in as google cloud asset in google earth engine

10. <b>results.ipynb</b> - visualize all the modle performances across 172 experiments

11. <b>pytest.ipynb</b> - Although, some pytests were already automatically integrated with github, namely the `config.py`, `losses.py`, `metrics_.py` some tests were not possible on the apple mac M1 machine. This is because of the tensorflow installation issue, the need to pay virtual machine to connect to google cloud bucket and authorization of the earth engine module. Hence, running pytest on google colab is suitable.



#### For Folders:

Inside the <b>'.github/workflows'</b> folder there are four .yml files which consists of code that will automate PEP8 testing (both for .py and .ipynb notebooks), pytests and sphinx documentation everytime there is a a push to the repository.

Inside the <b>'docs'</b> folder there is a document for how to use the Pytest cases inside the 'tools/tests' folder or open the html/index.html on local computer. 

Inside the <b>'tools'</b> folder there are the functions that were imported and used throughout our notebooks. They are under the names 'preprocessing.py', 'model.py', 'losses_.py', 'sampling.py', 'metrics_.py' and 'config.py'. Also, inside the 'tools' folder there is also a folder named 'tests' which enables doing pytests on the functions within the 'tools' folder.





### Important Note
We mainly used Google Colab to write the notebook rather than in local machine due to memory issues.  

### Documentation
You can find the documentation for how to use the functions inside the 'tools' folder inside docs/waterclassification.pdf or open the html/index.html on your local computer. 


# IRP

<p align="center">
  <img src="https://user-images.githubusercontent.com/90031508/183531098-494a5819-7714-4f72-8ff8-d038982eb5f0.png" alt="Water Oracle logo"/>
</p>

## Combining Landsat-8, Sentinel-1 and NASADEM using Multi-view Learning to Classify Water Bodies Globally

A Repository for independent research project. A multi-view deep learning approach with a U-Net core is developed and used to fuse information from different satellites (Landsat-8, L8SR; Sentinel-1, S1) and a digital elevation model (NASDEM)
to identify surface water bodies globally. 

Read more about the project at: https://github.com/ese-msc-2021/irp-kl121/blob/main/reports/kl121-final-report.pdf

## Important Note

⚠️ The main purpose of this repository is to determine the best model for water bodies prediction by training 154 variations of U-Net models. The best model is called the WaterNet and applications of WaterNet can be found in a React-built website called WaterOracle accesible at https://geeimperial.herokuapp.com/ or the public repository https://github.com/edsml-kl121/geeimperial.


## Prerequisites
Before running the notebooks the following must be done:

- Sign up for a google account and the account will be used for authentication everytime you use the Google Collaboratory (GC) notebook.
- (Optional) The experiments in this repository used GC Pro+ to help with long runtime so subscription to GC pro+ is recommended. 
- Sign up for a Google cloud platform (GCP) account so that you can use the Google Cloud Bucket (GCB). You must retrieve the project id of your project and set it at the start (storing training samples over 15GB will lead to extra charges)
- Get the private key to your GCP project at https://developers.google.com/earth-engine/guides/service_account.
- Sign up for a Wandb.ai account that is free of charge. Wandb.ai enables tracking different metrics such as precision, recall and F1 score of all the experiments.


## The Workflow
There are in total of 11 main notebooks serving different functions and in order to run these notebooks you must install the prequisites.


| ID | Notebook | Description |
| :--- | :--- | :--- |
1 | **[Preprocessing_and_export.ipynb](https://colab.research.google.com/drive/1VHBIUorm3GaDxb_GQFhRb-WpSfokexAO?usp=sharing)** | Exporting training and testing data in Thailand to google cloud bucket for training models in the other notebook `TrainModels.ipynb` and also for accessing the performance in `metrics_assessment.ipynb`|
2 | **[TrainingModels.ipynb](https://colab.research.google.com/drive/1My8P6hB8Ej8VhVpSOeUEDVeJRlLY9u3G?usp=sharing)** | Train the models using data stored in Google Cloud Bucket (GCB) from `Preprocessing_and_export.ipynb` notebook. The training is accompanied with Wandb and is splitted into three different methods of training featurestack, multiview learning with 2 and 3 inputs variation of UNET.|
3 | **[metrics_assessment.ipynb](https://colab.research.google.com/drive/1bnlsNuiwyNLvr-Z84BowAwUAKxEGhkjo?usp=sharing)** | Evaluate the performance of our trained neural network and NDWI on different metrics including recall, precision, F1 score and accuracy.|
4 | **[Preprocessing_and_export_global.ipynb](https://colab.research.google.com/drive/1FqVci8loCj-C0Zi-SynCM4IOqu1b_ihJ?usp=sharing)** | Exporting global training data to google cloud bucket by using earth engine's package to export multiple training patches that will be used to train models in the other notebook (TrainModels_global.ipynb)|
5 | **[TrainingModels_global.ipynb](https://colab.research.google.com/drive/1vQd4gM12G3LD51H-7qSUgtgs04dX3xCF?usp=sharing)** | Train the models that has been stored in the google cloud buckets by the `Preprocessing_and_export_global.ipynb` notebook. The training is accompanied with Wandb and is splitted into three different methods of training featurestack, multiview learning with 2 and 3 inputs variation of UNET.|
6 | **[metrics_assessment_global.ipynb](https://colab.research.google.com/drive/1yLEomdTejdg9eXtTaMvWWJGWb2Qj0iKj?usp=sharing)** | Evaluate the performance of our trained neural network on different metrics including recall, precision, F1 score and accuracy. The models have been created in `TrainModels_global.ipynb` and stored in the google cloud bucket. The metrics evaluation is accompanied with Wandb. Here we will evaluate 3 different performances Feature stack on the tests data in 10 different locations, Multiview learning with 2 inputs on the tests data in 10 different locations and Multiview learning with 3 inputs on the tests data in 10 different locations.|
7 | **[Hyperparameter_tuning.ipynb.ipynb](https://colab.research.google.com/drive/1qAlYJH1zNVuOTPBG2AzsK1P0JxPN9uMc?usp=sharing)** | Hypermeter tuning on the loss functions of the WaterNet trained in Thailand.|
8 | **[Hyperparameter_tuning_global.ipynb](https://colab.research.google.com/drive/1e9ojgezbOBYoCxJ2HS9dtCTc9_8-uOZt?usp=sharing)** | Hypermeter tuning on the loss functions of the WaterNet trained in globally.|
9 | **[ImageExport.ipynb](https://colab.research.google.com/drive/1sXx5bcDqL6P1S6Z0ZYpEOmC8cTnd4DPz?usp=sharing)** | Export image in regions of interest and predict this image and generate the output in as google cloud asset in google earth engine|
10 | **[results.ipynb](https://github.com/ese-msc-2021/irp-kl121/blob/main/results.ipynb)** | visualize all the modle performances across experiments|
11 | **[pytest.ipynb](https://colab.research.google.com/drive/1cRvhjD0407YfCaQDA-Y_pKN496YnsV0T?usp=sharing)** | Manual pytest on Google Collaboratory. Some pytests were already automatically integrated with github, namely the `config.py`, `losses.py`, `metrics_.py`,but some tests were not possible on the apple mac M1 machine. This is because of the tensorflow installation issue and GEE authorization issue.




- The notebook 1,2,3 and 4,5,6 should be run in chronological order. 
- Notebook 7 and 8 is used to tune the best models. 
- To visualize the prediction use notebook 9.
- Notebook 10 is used to analyse the performance of our model by evaluating its prediction.  
- Additionally the CloudcoverExp.Rmd is used to perform statistical analysis on cloud cover and our models.
- Test the custom tool package use notebook 11. 


## Getting started guide

### Case 1: Running in Google Collaboratory (recommended)

- Modify the project ID, and the private key from the current repository to yours
- Change the Google Cloud Bucket folder name
- Installing custom packages: There are three ways to import custom packages to the GC notebook. First, at the start of each notebook there are cells to create custom functions from tools folder and these cells should be run in order. Second, users can manually upload the files in tools folder into GC. Thirdly, users can clone the repository. The third method is not recommended because irp-kl121 is a private repository and you may encounter permission issues.
```
git clone git@github.com:ese-msc-2021/irp-kl121.git
```
- Now you can run the notebook in order :)

### Case 2: Local Machine (Not recommended)


Since the workflow involves integration with GCB, if users want to run the tests locally, they should follow https://research.google.com/colaboratory/local-runtimes.html and connect the notebook to Google Compute Engine to increase GPU unless the user's local machines have high enough GPU. Then, the user should clone the repository:
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


### Testing

Pytests are split into manual and automatic testing because of the local mac M1 chip difficulty in tensorflow installation and the difficulty in communication with google cloud bucket. The tests are splitted into automatic testing and manual testing.

Automatic testing are in tools/tests folder and includes `test_config.py`, `test_metrics_.py` and `test_preprocessing.py`
Manual testing are in tools/test_manual folder and includes `test_images.py`, `test_losses_.py`, `test_metrics.py`, `test_model.py`, `test_preprocessing.py`, `test_sampling.py`

For automated github pytest workflow we can do
```
pytest tools/tests/
```
For manual pytest, run the pytest.ipynb in GC, but you would need to install the <b>prequisites</b>.


#### For Folders:

Inside the <b>'.github/workflows'</b> folder there are four .yml files which consists of code that will automate PEP8 testing (both for .py and .ipynb notebooks), pytests and sphinx documentation everytime there is a a push to the repository.

Inside the <b>'docs'</b> folder there is a document for how to use our custom tools package.
The Pytest cases inside the 'tools/tests' folder or open the html/index.html on local computer. 

Inside the <b>'tools'</b> folder there are the functions that were imported and used throughout our notebooks. They are under the names 'preprocessing.py', 'model.py', 'losses_.py', 'sampling.py', 'metrics_.py' and 'config.py'. Also, inside the 'tools' folder there is also a folder named 'tests' which enables doing pytests on the functions within the 'tools' folder.


### Important Note
We mainly used Google Colab to write the notebook rather than in local machine due to memory issues.  

### Documentation
You can find the documentation for how to use the functions inside the 'tools' folder inside docs/waterclassification.pdf or open the html/index.html on your local computer. 

<!-- LICENSE -->
## License
Distributed under the Apache License. See `license.md` for more information.

<!-- CONTACT -->
## Contact
Kandanai Leenutaphong - kl121@ic.ac.uk


Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/NotebookVM/how-to-use-azureml/automated-machine-learning/classification-credit-card-fraud/auto-ml-classification-credit-card-fraud.png)

# Automated Machine Learning
_**Classification with Deployment using Credit Card Dataset**_

## Contents
1. [Introduction](#Introduction)
1. [Setup](#Setup)
1. [Train](#Train)
1. [Results](#Results)
1. [Deploy](#Deploy)
1. [Test](#Test)
1. [Acknowledgements](#Acknowledgements)

## Introduction

In this example we use the associated credit card dataset to showcase how you can use AutoML for a simple classification problem and deploy it to an Azure Container Instance (ACI). The classification goal is to predict if a creditcard transaction  is or is not considered a fraudulent charge.

If you are using an Azure Machine Learning Notebook VM, you are all set.  Otherwise, go through the [configuration](../../../configuration.ipynb)  notebook first if you haven't already to establish your connection to the AzureML Workspace. 

In this notebook you will learn how to:
1. Create an experiment using an existing workspace.
2. Configure AutoML using `AutoMLConfig`.
3. Train the model using local compute.
4. Explore the results.
5. Register the model.
6. Create a container image.
7. Create an Azure Container Instance (ACI) service.
8. Test the ACI service.

## Setup

As part of the setup you have already created an Azure ML `Workspace` object. For AutoML you will need to create an `Experiment` object, which is a named object in a `Workspace` used to run experiments.


```python
import logging

from matplotlib import pyplot as plt
import pandas as pd
import os

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
```


```python
ws = Workspace.from_config()

# choose a name for experiment
experiment_name = 'automl-classification-ccard'

experiment=Experiment(ws, experiment_name)

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SDK version</th>
      <td>1.0.65</td>
    </tr>
    <tr>
      <th>Subscription ID</th>
      <td>5e28df68-f866-4340-af02-dd7b4502de0b</td>
    </tr>
    <tr>
      <th>Workspace</th>
      <td>mot-amls-dev</td>
    </tr>
    <tr>
      <th>Resource Group</th>
      <td>mot-dev-rg</td>
    </tr>
    <tr>
      <th>Location</th>
      <td>uksouth</td>
    </tr>
    <tr>
      <th>Experiment Name</th>
      <td>automl-classification-ccard</td>
    </tr>
  </tbody>
</table>
</div>



## Create or Attach existing AmlCompute
You will need to create a compute target for your AutoML run. In this tutorial, you create AmlCompute as your training compute resource.
#### Creation of AmlCompute takes approximately 5 minutes. 
If the AmlCompute with that name is already in your workspace this code will skip the creation process.
As with other Azure services, there are limits on certain resources (e.g. AmlCompute) associated with the Azure Machine Learning service. Please read this article on the default limits and how to request more quota.


```python
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

# Choose a name for your cluster.
amlcompute_cluster_name = "automlcl"

found = False
# Check if this compute target already exists in the workspace.
cts = ws.compute_targets
if amlcompute_cluster_name in cts and cts[amlcompute_cluster_name].type == 'AmlCompute':
    found = True
    print('Found existing compute target.')
    compute_target = cts[amlcompute_cluster_name]
    
if not found:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2", # for GPU, use "STANDARD_NC6"
                                                                #vm_priority = 'lowpriority', # optional
                                                                max_nodes = 6)

    # Create the cluster.
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, provisioning_config)
    
print('Checking cluster status...')
# Can poll for a minimum number of nodes and for a specific timeout.
# If no min_node_count is provided, it will use the scale settings for the cluster.
compute_target.wait_for_completion(show_output = True, min_node_count = None, timeout_in_minutes = 20)

# For a more detailed view of current AmlCompute status, use get_status().
```

    Found existing compute target.
    Checking cluster status...
    Succeeded
    AmlCompute wait for completion finished
    Minimum number of nodes requested have been provisioned


# Data

Create a run configuration for the remote run.


```python
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
import pkg_resources

# create a new RunConfig object
conda_run_config = RunConfiguration(framework="python")

# Set compute target to AmlCompute
conda_run_config.target = compute_target
conda_run_config.environment.docker.enabled = True

cd = CondaDependencies.create(conda_packages=['numpy','py-xgboost<=0.80'])
conda_run_config.environment.python.conda_dependencies = cd
```

### Load Data

Load the credit card dataset from a csv file containing both training features and labels. The features are inputs to the model, while the training labels represent the expected output of the model. Next, we'll split the data using random_split and extract the training data for the model.


```python
data = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/creditcard.csv"
dataset = Dataset.Tabular.from_delimited_files(data)
training_data, validation_data = dataset.random_split(percentage=0.8, seed=223)
label_column_name = 'Class'
X_test = validation_data.drop_columns(columns=[label_column_name])
y_test = validation_data.keep_columns(columns=[label_column_name], validate=True)

```

## Train

Instantiate a AutoMLConfig object. This defines the settings and data used to run the experiment.

|Property|Description|
|-|-|
|**task**|classification or regression|
|**primary_metric**|This is the metric that you want to optimize. Classification supports the following primary metrics: <br><i>accuracy</i><br><i>AUC_weighted</i><br><i>average_precision_score_weighted</i><br><i>norm_macro_recall</i><br><i>precision_score_weighted</i>|
|**iteration_timeout_minutes**|Time limit in minutes for each iteration.|
|**iterations**|Number of iterations. In each iteration AutoML trains a specific pipeline with the data.|
|**n_cross_validations**|Number of cross validation splits.|
|**training_data**|Input dataset, containing both features and label column.|
|**label_column_name**|The name of the label column.|

**_You can find more information about primary metrics_** [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train#primary-metric)

##### If you would like to see even better results increase "iteration_time_out minutes" to 10+ mins and increase "iterations" to a minimum of 30


```python
automl_settings = {
    "iteration_timeout_minutes": 5,
    "iterations": 10,
    "n_cross_validations": 2,
    "primary_metric": 'average_precision_score_weighted',
    "preprocess": True,
    "max_concurrent_iterations": 5,
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             run_configuration=conda_run_config,
                             training_data = training_data,
                             label_column_name = label_column_name,
                             **automl_settings
                            )
```

Call the `submit` method on the experiment object and pass the run configuration. Execution of local runs is synchronous. Depending on the data and the number of iterations this can run for a while.
In this example, we specify `show_output = True` to print currently running iterations to the console.


```python
remote_run = experiment.submit(automl_config, show_output = True)
```

    Running on remote compute: automlcl
    Parent Run ID: AutoML_6dc5fe66-ce16-4404-a034-e9068186489e
    Current status: FeaturesGeneration. Generating features for the dataset.
    Current status: DatasetCrossValidationSplit. Generating individually featurized CV splits.
    Current status: ModelSelection. Beginning model selection.
    
    ****************************************************************************************************
    ITERATION: The iteration being evaluated.
    PIPELINE: A summary description of the pipeline being evaluated.
    DURATION: Time taken for the current iteration.
    METRIC: The result of computing score on the fitted pipeline.
    BEST: The best observed score thus far.
    ****************************************************************************************************
    
     ITERATION   PIPELINE                                       DURATION      METRIC      BEST
             2   StandardScalerWrapper SGD                      0:01:13       0.7333    0.7333
             4   StandardScalerWrapper SGD                      0:02:13       0.7227    0.7333
             3   MinMaxScaler LightGBM                          0:03:23       0.7923    0.7923
             7   MinMaxScaler SGD                               0:01:15       0.6305    0.7923
             5   StandardScalerWrapper LightGBM                 0:06:49       0.8064    0.8064
             1   StandardScalerWrapper ExtremeRandomTrees       0:08:08       0.6279    0.8064
             6   MinMaxScaler LightGBM                          0:05:52       0.8069    0.8069
             0   StandardScalerWrapper SGD                      0:08:17       0.2138    0.8069
             8    VotingEnsemble                                0:01:28       0.8150    0.8150
             9    StackEnsemble                                 0:01:52       0.8119    0.8150



```python
remote_run
```




<table style="width:100%"><tr><th>Experiment</th><th>Id</th><th>Type</th><th>Status</th><th>Details Page</th><th>Docs Page</th></tr><tr><td>automl-classification-ccard</td><td>AutoML_6dc5fe66-ce16-4404-a034-e9068186489e</td><td>automl</td><td>Completed</td><td><a href="https://mlworkspace.azure.ai/portal/subscriptions/5e28df68-f866-4340-af02-dd7b4502de0b/resourceGroups/mot-dev-rg/providers/Microsoft.MachineLearningServices/workspaces/mot-amls-dev/experiments/automl-classification-ccard/runs/AutoML_6dc5fe66-ce16-4404-a034-e9068186489e" target="_blank" rel="noopener">Link to Azure Portal</a></td><td><a href="https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py" target="_blank" rel="noopener">Link to Documentation</a></td></tr></table>



## Results

#### Widget for Monitoring Runs

The widget will first report a "loading" status while running the first iteration. After completing the first iteration, an auto-updating graph and table will be shown. The widget will refresh once per minute, so you should see the graph update as child runs complete.

**Note:** The widget displays a link at the bottom. Use this link to open a web interface to explore the individual run details


```python
from azureml.widgets import RunDetails
RunDetails(remote_run).show() 
```


    A Jupyter Widget


## Deploy

### Retrieve the Best Model

Below we select the best pipeline from our iterations. The `get_output` method on `automl_classifier` returns the best run and the fitted model for the last invocation. Overloads on `get_output` allow you to retrieve the best run and fitted model for *any* logged metric or for a particular *iteration*.


```python
best_run, fitted_model = remote_run.get_output()
```

### Register the Fitted Model for Deployment
If neither `metric` nor `iteration` are specified in the `register_model` call, the iteration with the best primary metric is registered.


```python
description = 'AutoML Model'
tags = None
model = remote_run.register_model(description = description, tags = tags)

print(remote_run.model_id) # This will be written to the script file later in the notebook.
```

    AutoML636c65f01best


### Create Scoring Script
The scoring script is required to generate the image for deployment. It contains the code to do the predictions on input data.


```python
%%writefile score.py
import pickle
import json
import numpy
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path(model_name = '<<modelid>>') # this name is model.id of model that we want to deploy
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

def run(rawdata):
    try:
        data = json.loads(rawdata)['data']
        data = numpy.array(data)
        result = model.predict(data)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result":result.tolist()})
```

    Writing score.py


### Create a YAML File for the Environment

To ensure the fit results are consistent with the training results, the SDK dependency versions need to be the same as the environment that trains the model. Details about retrieving the versions can be found in notebook [12.auto-ml-retrieve-the-training-sdk-versions](12.auto-ml-retrieve-the-training-sdk-versions.ipynb).


```python
dependencies = remote_run.get_run_sdk_dependencies(iteration = 1)
```

    WARNING - The version of the SDK does not match the version the model was trained on.
    WARNING - The consistency in the result may not be guaranteed.
    WARNING - Package:azureml-dataprep-native, training version:13.1.0, current version:13.0.3



```python
for p in ['azureml-train-automl', 'azureml-core']:
    print('{}\t{}'.format(p, dependencies[p]))
```

    azureml-train-automl	1.0.65
    azureml-core	1.0.65.1



```python
myenv = CondaDependencies.create(conda_packages=['numpy','scikit-learn','py-xgboost<=0.80'],
                                 pip_packages=['azureml-defaults','azureml-train-automl'])

conda_env_file_name = 'myenv.yml'
myenv.save_to_file('.', conda_env_file_name)
```




    'myenv.yml'




```python
# Substitute the actual version number in the environment file.
# This is not strictly needed in this notebook because the model should have been generated using the current SDK version.
# However, we include this in case this code is used on an experiment from a previous SDK version.

with open(conda_env_file_name, 'r') as cefr:
    content = cefr.read()

with open(conda_env_file_name, 'w') as cefw:
    cefw.write(content.replace(azureml.core.VERSION, dependencies['azureml-train-automl']))

# Substitute the actual model id in the script file.

script_file_name = 'score.py'

with open(script_file_name, 'r') as cefr:
    content = cefr.read()

with open(script_file_name, 'w') as cefw:
    cefw.write(content.replace('<<modelid>>', remote_run.model_id))
```

### Deploy the model as a Web Service on Azure Container Instance


```python
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import Model

inference_config = InferenceConfig(runtime = "python", 
                                   entry_script = script_file_name,
                                   conda_file = conda_env_file_name)

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {'area': "cards", 'type': "automl_classification"}, 
                                               description = 'sample service for Automl Classification')

aci_service_name = 'automl-sample-creditcard'
print(aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(aci_service.state)
```

    automl-sample-creditcard
    Running......................................................................................................................
    SucceededACI service creation operation finished, operation "Succeeded"
    Healthy


### Delete a Web Service

Deletes the specified web service.


```python
#aci_service.delete()
```

### Get Logs from a Deployed Web Service

Gets logs from a deployed web service.


```python
#aci_service.get_logs()
```

## Test

Now that the model is trained, split the data in the same way the data was split for training (The difference here is the data is being split locally) and then run the test data through the trained model to get the predicted values.


```python
#Randomly select and test
X_test = X_test.to_pandas_dataframe()
y_test = y_test.to_pandas_dataframe()

```


```python
y_pred = fitted_model.predict(X_test)
y_pred
```

### Calculate metrics for the prediction

Now visualize the data on a scatter plot to show what our truth (actual) values are compared to the predicted values 
from the trained model that was returned.


```python
#Randomly select and test
# Plot outputs
%matplotlib notebook
test_pred = plt.scatter(y_test, y_pred, color='b')
test_test = plt.scatter(y_test, y_test, color='g')
plt.legend((test_pred, test_test), ('prediction', 'truth'), loc='upper left', fontsize=8)
plt.show()


```

## Acknowledgements

This Credit Card fraud Detection dataset is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in individual contents of the database are licensed under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/ and is available at: https://www.kaggle.com/mlg-ulb/creditcardfraud


The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (UniversitÃ© Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project
Please cite the following works: 
â€¢	Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
â€¢	Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon
â€¢	Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE
o	Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)
â€¢	Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-AÃ«l; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier
â€¢	Carcillo, Fabrizio; Le Borgne, Yann-AÃ«l; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

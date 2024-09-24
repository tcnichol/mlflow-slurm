from dotenv import load_dotenv

load_dotenv()

#!/usr/bin/env python
# coding: utf-8

# ## MLflow 5 minute Tracking Quickstart

# This notebook demonstrates using a local MLflow Tracking Server to log, register, and then load a model as a generic Python Function (pyfunc) to perform inference on a Pandas DataFrame.
#
# Throughout this notebook, we'll be using the MLflow fluent API to perform all interactions with the MLflow Tracking Server.

# In[ ]:


import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature


# ### Set the MLflow Tracking URI
#
# Depending on where you are running this notebook, your configuration may vary for how you initialize the interface with the MLflow Tracking Server.
#
# For this example, we're using a locally running tracking server, but other options are available (The easiest is to use the free managed service within [Databricks Community Edition](https://community.cloud.databricks.com/)).
#
# Please see [the guide to running notebooks here](https://www.mlflow.org/docs/latest/getting-started/running-notebooks/index.html) for more information on setting the tracking server uri and configuring access to either managed or self-managed MLflow tracking servers.

# In[2]:


# NOTE: review the links mentioned above for guidance on connecting to a managed tracking server, such as the free Databricks Community Edition

mlflow.set_tracking_uri(uri="https://mlflow.margindx.software-dev.ncsa.illinois.edu")


# ## Load training data and train a simple model
#
# For our quickstart, we're going to be using the familiar iris dataset that is included in scikit-learn. Following the split of the data, we're going to train a simple logistic regression classifier on the training data and calculate some error metrics on our holdout test data.
#
# Note that the only MLflow-related activities in this portion are around the fact that we're using a `param` dictionary to supply our model's hyperparameters; this is to make logging these settings easier when we're ready to log our model and its associated metadata.

# In[3]:


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate accuracy as a target loss metric
accuracy = accuracy_score(y_test, y_pred)


# ## Define an MLflow Experiment
#
# In order to group any distinct runs of a particular project or idea together, we can define an Experiment that will group each iteration (runs) together.
# Defining a unique name that is relevant to what we're working on helps with organization and reduces the amount of work (searching) to find our runs later on.

# In[4]:


mlflow.set_experiment("MLflow Quickstart 3")


# ## Log the model, hyperparameters, and loss metrics to MLflow.
#
# In order to record our model and the hyperparameters that were used when fitting the model, as well as the metrics associated with validating the fit model upon holdout data, we initiate a run context, as shown below. Within the scope of that context, any fluent API that we call (such as `mlflow.log_params()` or `mlflow.sklearn.log_model()`) will be associated and logged together to the same run.

# In[9]:


from datetime import datetime

# Generate a timestamp for the run name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"Run_{timestamp}"


# In[10]:


# Start an MLflow run
with mlflow.start_run(run_name=run_name):
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )


# ## Load our saved model as a Python Function
#
# Although we can load our model back as a native scikit-learn format with `mlflow.sklearn.load_model()`, below we are loading the model as a generic Python Function, which is how this model would be loaded for online model serving. We can still use the `pyfunc` representation for batch use cases, though, as is shown below.

# In[7]:


loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)


# ## Use our model to predict the iris class type on a Pandas DataFrame

# In[8]:


predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

# Convert X_test validation feature data to a Pandas DataFrame
result = pd.DataFrame(X_test, columns=iris_feature_names)

# Add the actual classes to the DataFrame
result["actual_class"] = y_test

# Add the model predictions to the DataFrame
result["predicted_class"] = predictions

result[:4]


# In[ ]:





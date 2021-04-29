# Databricks notebook source
# MAGIC %md
# MAGIC ## Distributed hyperparameter tuning with Hyperopt + SparkTrials
# MAGIC 
# MAGIC - Dataset: [Credit Card Fraud Detection, Worldline & Machine Learning Group of ULB](http://mlg.ulb.ac.be/)
# MAGIC   - Anonymized credit card transactions labeled as fraudulent or genuine
# MAGIC   - Hosted on Databricks at `/databricks-datasets/credit-card-fraud`
# MAGIC 
# MAGIC - Model: `sklearn.ensemble.RandomForestClassifier`
# MAGIC - Tuning: Hyperopt with `SparkTrials`
# MAGIC   - An open-source tuning library that employs Tree of Parzien Estimators (TPE)
# MAGIC   - Spark backend for distributed tuning

# COMMAND ----------

import mlflow
import pandas as pd

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials, SparkTrials
from hyperopt.pyll.base import scope

from pyspark.ml.functions import vector_to_array

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

mlflow.sklearn.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load and prepare the data

# COMMAND ----------

spark_df = spark.read.parquet("/databricks-datasets/credit-card-fraud/data")
spark_df = spark_df.withColumn("pca", vector_to_array("pcaVector"))

pca_length = 28
pca_columns = [spark_df.pca[i] for i in range(pca_length)]

data = spark_df.select(["time", "amountRange", "label"] + pca_columns).toPandas()
data = data.fillna(data.mean())
target_col = "label"

# COMMAND ----------

data.head()

# COMMAND ----------

# Split into train and validation sets
X = data.drop([target_col], axis=1)
y = data[target_col]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single-node hyperparameter tuning

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define search space and objective function

# COMMAND ----------

# Define the hyperparameter search space
space = {
  "n_estimators": scope.int(hp.quniform("n_estimators", 50, 150, 1)),
  "criterion": hp.choice("criterion", ["gini", "entropy"]),
  "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
  "min_samples_split": hp.uniform("min_samples_split", 0, 1),
}

n_evals = 200

# COMMAND ----------

def objective(hyperparameters):
  # Instantiate the model with hyperparameters
  model = RandomForestClassifier(**hyperparameters)

  # Train the model
  model.fit(X_train, y_train)

  # Evaluate the learned model
  val_pred = model.predict(X_val)
  val_f1_score = f1_score(y_val, val_pred)

  # Use negative F1 score as our loss metric
  return {"loss": -val_f1_score, "status": STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train a `RandomForestClassifier` with Hyperopt

# COMMAND ----------

trials = Trials()
best = fmin(
  fn=objective,
  space=space,
  algo=tpe.suggest,
  max_evals=n_evals,
  trials=trials
)

best_f1 = sorted(trials.results, key=lambda result: result["loss"])[0]["loss"] * -1

# COMMAND ----------

best["F1 Score"] = best_f1
best["criterion"] = "gini" if best["criterion"] == 0 else "entropy"
display(pd.DataFrame(best, index=[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed hyperparameter tuning

# COMMAND ----------

# MAGIC %md
# MAGIC #### Broadcast data

# COMMAND ----------

# Broadcast the data
X_train_bc = sc.broadcast(X_train)
y_train_bc = sc.broadcast(y_train)
X_val_bc = sc.broadcast(X_val)
y_val_bc = sc.broadcast(y_val)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train a `RandomForestClassifier` with Hyperopt + `SparkTrials`

# COMMAND ----------

def objective(hyperparameters):
  # Get the broadcasted variables
  X_train = X_train_bc.value
  y_train = y_train_bc.value
  X_val = X_val_bc.value
  y_val = y_val_bc.value
  
  # Instantiate the model with hyperparameters
  model = RandomForestClassifier(**hyperparameters)

  # Train the model
  model.fit(X_train, y_train)

  # Evaluate the learned model
  val_pred = model.predict(X_val)
  val_f1_score = f1_score(y_val, val_pred)

  # Use negative F1 score as our loss metric
  return {"loss": -val_f1_score, "status": STATUS_OK}

# COMMAND ----------

trials = SparkTrials(parallelism=32)
best = fmin(
  fn=objective,
  space=space,
  algo=tpe.suggest,
  max_evals=n_evals,
  trials=trials
)

best_f1 = sorted(trials.results, key=lambda result: result["loss"])[0]["loss"] * -1

# COMMAND ----------

best["F1 Score"] = best_f1
best["criterion"] = "gini" if best["criterion"] == 0 else "entropy"
display(pd.DataFrame(best, index=[0]))

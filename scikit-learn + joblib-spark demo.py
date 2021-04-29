# Databricks notebook source
# MAGIC %md
# MAGIC ## Distributed hyperparameter tuning with scikit-learn + joblib-spark
# MAGIC 
# MAGIC - Dataset: [Credit Card Fraud Detection, Worldline & Machine Learning Group of ULB](http://mlg.ulb.ac.be/)
# MAGIC   - Anonymized credit card transactions labeled as fraudulent or genuine
# MAGIC   - Hosted on Databricks at `/databricks-datasets/credit-card-fraud`
# MAGIC 
# MAGIC - Model: `sklearn.ensemble.RandomForestClassifier`
# MAGIC - Tuning: `sklearn.model_selection.RandomizedSearchCV`
# MAGIC   - scikit-learn's random search implementation with cross validation
# MAGIC   - Uses `joblib-spark` as a backend for distributed tuning

# COMMAND ----------

import mlflow
import pandas as pd
import scipy.stats as stats

from joblibspark import register_spark

from pyspark.ml.functions import vector_to_array

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import parallel_backend

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

X = data.drop([target_col], axis=1)
y = data[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define the search space

# COMMAND ----------

# Define the hyperparameter search space
space = {
  "n_estimators": stats.randint(50, 150),
  "criterion": ["gini", "entropy"],
  "min_samples_leaf": stats.randint(1, 20),
  "min_samples_split": stats.uniform(0, 1)
}

n_evals = 200

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single-node hyperparameter tuning

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train a `RandomForestClassifier` with `RandomizedSearchCV`

# COMMAND ----------

model = RandomForestClassifier()

search = RandomizedSearchCV(
  estimator=model,
  param_distributions=space,
  n_iter=n_evals,
  n_jobs=8,
  cv=2,
  verbose=2
)
search.fit(X, y)

# COMMAND ----------

best = search.best_params_
best["CV Score"] = search.best_score_
display(pd.DataFrame(best, index=[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed hyperparameter tuning

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train a `RandomForestClassifier` with `RandomizedSearchCV` + `joblib-spark`

# COMMAND ----------

register_spark()

parallelism = 64
with parallel_backend("spark", n_jobs=parallelism):
  search = RandomizedSearchCV(
    estimator=model,
    param_distributions=space,
    n_iter=n_evals,
    cv=2,
    verbose=2
  )
  search.fit(X, y)

# COMMAND ----------

best = search.best_params_
best["CV Score"] = search.best_score_
display(pd.DataFrame(best, index=[0]))

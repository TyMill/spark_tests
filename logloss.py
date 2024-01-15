from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log
from pyspark.sql.types import DoubleType
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder.appName("GBTClassifierLogLoss").getOrCreate()

# Assuming 'df' is your DataFrame with features and 'label' as the target column
# Convert features to a vector
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
data = assembler.transform(df)

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3])

# Train the model
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
model = gbt.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Calculate Log Loss
# Note: GBTClassifier does not output probability by default in PySpark. 
# If your version does, use the 'probability' column instead of 'rawPrediction'.
epsilon = 1e-15
def compute_log_loss(prob, label):
    prob = np.clip(prob, epsilon, 1 - epsilon)
    return -(label * np.log(prob) + (1 - label) * np.log(1 - prob))

compute_log_loss_udf = udf(compute_log_loss, DoubleType())

predictions = predictions.withColumn("log_loss", compute_log_loss_udf(col("probability"), col("label")))
avg_log_loss = predictions.groupBy().avg("log_loss").collect()[0][0]

print("Average Log Loss: ", avg_log_loss)

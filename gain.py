from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt

# Initialize Spark Session
spark = SparkSession.builder.appName("GBTClassifierExample").getOrCreate()
# Assuming 'df' is your DataFrame with features and 'label' as the target column
# Convert features to a vector
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
data = assembler.transform(df)

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3])

# Create and train the GBT model
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
model = gbt.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Convert Spark DataFrame to Pandas DataFrame for easier manipulation
predictions_pd = predictions.select("probability", "label").toPandas()

# Extract the probability for the positive class
predictions_pd['prob'] = predictions_pd['probability'].apply(lambda x: x[1])

# Sort by probability in descending order
predictions_pd = predictions_pd.sort_values(by='prob', ascending=False)

# Calculate the cumulative number of positive cases
predictions_pd['cumulative_positives'] = predictions_pd['label'].cumsum()

# Normalize to get the gain
predictions_pd['gain'] = predictions_pd['cumulative_positives'] / predictions_pd['label'].sum()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(predictions_pd['gain'], label='GBT Model')
plt.plot([0, 1], [0, 1], 'k--', label='Random Model')
plt.xlabel('Proportion of data')
plt.ylabel('Gain')
plt.title('Gain Chart')
plt.legend()
plt.show()

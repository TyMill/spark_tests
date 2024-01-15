from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt

# Initialize Spark Session
spark = SparkSession.builder.appName("GBTClassifierROC").getOrCreate()

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

# Evaluate the model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print("ROC AUC: ", roc_auc)

# Select the required columns
selected = predictions.select("probability", "label")

# Convert to Pandas DataFrame for computation
predictions_pd = selected.toPandas()
predictions_pd['score'] = predictions_pd['probability'].apply(lambda x: x[1])

# Calculate TPR and FPR
thresholds = sorted(predictions_pd['score'].unique(), reverse=True)
tpr = []
fpr = []

for thresh in thresholds:
    tp = len(predictions_pd[(predictions_pd['score'] > thresh) & (predictions_pd['label'] == 1)])
    fp = len(predictions_pd[(predictions_pd['score'] > thresh) & (predictions_pd['label'] == 0)])
    fn = len(predictions_pd[(predictions_pd['score'] <= thresh) & (predictions_pd['label'] == 1)])
    tn = len(predictions_pd[(predictions_pd['score'] <= thresh) & (predictions_pd['label'] == 0)])

    tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'GBT Model (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Save the plot as a PNG file
output_file_path = "gain_chart.png"
plt.savefig(output_file_path)


print(f"Gain chart saved as {output_file_path}")

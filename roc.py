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

# Confusion Matrix Calculation
tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
tn = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()

# Extract the counts from the Spark DataFrame
tp = cmx.filter(cmx.Metric == "True Positives").select("Count").collect()[0][0]
fp = cmx.filter(cmx.Metric == "False Positives").select("Count").collect()[0][0]
tn = cmx.filter(cmx.Metric == "True Negatives").select("Count").collect()[0][0]
fn = cmx.filter(cmx.Metric == "False Negatives").select("Count").collect()[0][0]


„cmx = spark.createDataFrame([
    ("True Positives", tp),
    ("False Positives", fp),
    ("True Negatives", tn),
    ("False Negatives", fn)
], ["Metric", "Count"])


# Calculate the metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0

metrics_df = spark.createDataFrame([
    ("Accuracy", accuracy),
    ("Precision", precision),
    ("Recall", recall),
    ("Specificity", specificity),
    ("F1 Score", f1_score),
    ("False Positive Rate", fpr),
    ("Negative Predictive Value", npv),
    ("False Discovery Rate", fdr),
    ("Miss Rate", miss_rate),
    ("Matthews Correlation Coefficient", mcc)
], ["Metric", "Value"])

metrics_df.show(truncate=False)


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

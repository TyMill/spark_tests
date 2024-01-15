from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd

# Make predictions
predictions = gbt_model.transform(test_data)

# Convert to Pandas DataFrame
predictions_pd = predictions.select("probability", "label").toPandas()

# Extract the probability of the positive class
predictions_pd['prob'] = predictions_pd['probability'].apply(lambda x: x[1])

# Compute Precision-Recall values
precision, recall, thresholds = precision_recall_curve(predictions_pd['label'], predictions_pd['prob'])


# Plotting Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

# Save the plot as a PNG file
output_file_path = "precision_recall_curve.png"
plt.savefig(output_file_path)

print(f"Precision-Recall curve saved as {output_file_path}")

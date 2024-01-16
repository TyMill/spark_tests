from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

# Convert DataFrame to RDD
predictionAndLabels = predictions.select(['prediction', 'label']).withColumn('label', col('label').cast('double')).rdd

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

# Statistics by class
labels = predictionAndLabels.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

# Weighted stats
print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F1 Score = %s" % metrics.weightedFMeasure())
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

# Count of each class
for label in sorted(labels):
    print("Class %s count = %s" % (label, predictionAndLabels.filter(lambda lp: lp.label == label).count()))

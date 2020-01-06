#!/usr/bin/env python
# coding: utf-8

import pyspark.sql.functions as F


from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Create a spark session/application 
spark = SparkSession.builder.appName('Inputdata').getOrCreate()

# Import train_hire_stats.csv as dataframe using the defined schema
schema = StructType(
    [
        StructField("Zone_ID", ByteType(), False),
        StructField("Date", TimestampType(), False),
        StructField("Hour_slot", ByteType(), False),
        StructField("Hire_count", ShortType(), False)
    ]
)
train_df = spark.read.format("csv").option("header", "true").option("delimiter", ",").schema(schema).load("data/train_hire_stats.csv")

train_df = train_df.withColumn('Day_of_the_week', 
                               (F.date_format(train_df["Date"], "u").cast(IntegerType())))

train_df = train_df.withColumn('Month', 
                               (F.date_format(train_df["Date"], "M").cast(IntegerType())))

# Import test_hire_stats.csv as dataframe using the defined schema
schema = StructType(
    [
        StructField("Test_ID", ShortType(), False),
        StructField("Zone_ID", ByteType(), False),
        StructField("Date", TimestampType(), False),
        StructField("Hour_slot", ByteType(), False),
        StructField("Hire_count", ByteType(), False)
    ]
)
test_df = spark.read.format("csv").option("header", "true").option("delimiter", ",").schema(schema).load("data/test_hire_stats.csv")

test_df = test_df.withColumn('Day_of_the_week', 
                             (F.date_format(test_df["Date"], "u").cast(IntegerType())))

test_df = test_df.withColumn('Month', 
                             (F.date_format(test_df["Date"], "M").cast(IntegerType())))


assembler = VectorAssembler(
    inputCols = ['Zone_ID', 'Hour_slot', 'Day_of_the_week', 'Month'],
    outputCol = 'features'
    )

trainData = assembler.transform(train_df)
testData = assembler.transform(test_df)

trainData = trainData.withColumn('label', trainData.Hire_count)
testData = testData.withColumn('label', testData.Hire_count)


#labelIndexer = StringIndexer(inputCol='Hire_count', outputCol='indexedLabel').fit(trainData)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=24).fit(trainData)


'''
rf = RandomForestClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures', numTrees=100)
rf = rf.setImpurity('Gini')
rf = rf.setMaxDepth(10)
rf = rf.setFeatureSubsetStrategy('auto')
'''

rf = RandomForestRegressor(featuresCol = 'indexedFeatures')
rf = rf.setMaxDepth(11)
rf = rf.setNumTrees(101)

pipeline = Pipeline(stages=[featureIndexer, rf])

model = pipeline.fit(trainData)

predictions = model.transform(testData)
predictions = predictions.withColumn('Hire_count', predictions.prediction)

# Only run this when using trainingData to make predicition
'''
evaluator = RegressionEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'rmse')
rmse = evaluator.evaluate(predictions)
print(rmse)
'''

final_df = predictions.withColumn("Hire_count", predictions["prediction"].cast(IntegerType()))
final_df = final_df.select("Test_ID", "Zone_ID", "Date", "Hour_slot", "Hire_count").orderBy(F.asc("Test_ID"))
final_df = final_df.withColumn("Date", F.date_format(F.col("Date"), "yyyy-MM-dd"))
final_df.show()

# Write the modified dataframe to csv. 
## Spark write function will split the workload and save the output spread out over multiple parts
## Using cat and >  we will generate a single output file
final_df.write.mode("overwrite").csv('output/attempts/decisiontree-regressor101-5')
os.system('rm output/attempts/decisiontree-regressor101-5.csv')
os.system('cat output/attempts/decisiontree-regressor101-5/p* > output/attempts/decisiontree-regressor-101-5.csv')

#!/usr/bin/env python
# coding: utf-8

import pyspark.sql.functions as F

from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Create a spark session/application
spark = SparkSession.builder.appName('Exercise 1').getOrCreate()

# Import News_Final.csv as dataframe using the defined schema
schema = StructType(
    [
        StructField("IDLink", IntegerType(), True),
        StructField("Title", StringType(), True),
        StructField("Headline", StringType(), True),
        StructField("Source", StringType(), True),
        StructField("Topic", StringType(), True),
        StructField("PublishDate", TimestampType(), True),
        StructField("SentimentTitle", FloatType(), True),
        StructField("SentimentHeadline", FloatType(), True),
        StructField("Facebook", FloatType(), True),
        StructField("GooglePlus", FloatType(), True),
        StructField("LinkedIn", FloatType(), True)
    ]
)
df = spark.read.format("csv").option("header", "true").option("delimiter", ",").option('quote', '"').option('escape', '"').schema(schema).load("data/News_Final.csv")

# Create a new column 'SentimentTotal', this is the average of the SentimentTitle and SentimentHeadline
df = df.withColumn('SentimentTotal', (col('SentimentTitle') + col('SentimentHeadline')) / 2)

# Group by 'Topic' than aggregate with the sum and mean of 'SentimentTotal'
df = df.groupby('Topic').agg(F.sum('SentimentTotal'), F.mean('SentimentTotal'))

# Show time, print the table
df.show()

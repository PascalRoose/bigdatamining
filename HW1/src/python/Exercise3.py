#!/usr/bin/env python
# coding: utf-8

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Create a spark session/application
spark = SparkSession.builder.appName('Exercise 3').getOrCreate()

# Import household_power_consumption.txt as dataframe using the defined schema
schema = StructType(
    [
        StructField("Date", StringType(), True),
        StructField("Time", StringType(), True),
        StructField("Global_active_power", FloatType(), True),
        StructField("Global_reactive_power", FloatType(), True),
        StructField("Voltage", FloatType(), True),
        StructField("Global_intensity", FloatType(), True),
        StructField("Sub_metering_1", FloatType(), True),
        StructField("Sub_metering_2", FloatType(), True),
        StructField("Sub_metering_3", FloatType(), True)
    ]
)
df = spark.read.format("csv").option("header", "true").option("delimiter", ";").schema(schema).load("household_power_consumption.txt")

# Calculate the normalized values and add them to the dataframe as a new column
normalize = lambda xi, xmin, xmax : (xi - xmin) / (xmax - xmin)

columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
for column in columns:
    column_result = df.agg(
        min(column).alias('min'), 
        max(column).alias('max') 
    ).collect()[0]
    df = df.withColumn(column + '_norm', normalize(df[column], column_result['min'], column_result['max']))

df_norm = df.select([column + '_norm' for column in columns])

# Write the modified dataframe to csv. 
## Spark write function will split the workload and save the output spread out over multiple parts
## Using cat and >  we will generate a single output file
df_norm.write.mode("overwrite").csv('household_power_consumption_normalized')
os.system('rm household_power_consumption_normalized.txt')
os.system('cat household_power_consumption_normalized/p* > household_power_consumption_normalized.txt')


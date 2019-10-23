#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Create a spark session/application
spark = SparkSession.builder.appName('Exercise 1').getOrCreate()

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

# Calculate and print the min, max and count f
columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
for column in columns:
    print(f'-- {column} --')
    column_result = df.agg(
        min(column).alias('min'), 
        max(column).alias('max'), 
        count(column).alias('count')
    ).collect()[0]
    print(f"min: {column_result['min']}")
    print(f"max: {column_result['max']}")
    print(f"count: {column_result['count']}\n")


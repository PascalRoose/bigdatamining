#!/usr/bin/env python
# coding: utf-8

import pyspark.sql.functions as F

from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import *


# Create a spark session/application
spark = SparkSession.builder.appName('Term_Statistics').getOrCreate()

# Import train_hire_stats.csv as dataframe using the defined schema
schema = StructType(
    [
        StructField("Zone_ID", ByteType(), False),
        StructField("Date", StringType(), False),
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
        StructField("Date", StringType(), False),
        StructField("Hour_slot", ByteType(), False),
        StructField("Hire_count", ByteType(), False)
    ]
)
test_df = spark.read.format("csv").option("header", "true").option("delimiter", ",").schema(schema).load("data/test_hire_stats.csv")

test_df = test_df.withColumn('Day_of_the_week', 
                             (F.date_format(test_df["Date"], "u").cast(IntegerType())))

test_df = test_df.withColumn('Month', 
                             (F.date_format(test_df["Date"], "M").cast(IntegerType())))


compare_df = train_df.groupBy("Zone_ID", "Day_of_the_week", "Hour_slot").mean("Hire_count")
compare_df = compare_df.withColumn("avg(Hire_count)", compare_df["avg(Hire_count)"].cast(IntegerType()))


final_df = test_df.join(compare_df, ["Zone_ID", "Day_of_the_week", "Hour_slot"], "fullouter")
final_df = final_df.withColumn("Hire_count", final_df["avg(Hire_count)"])
final_df = final_df.select("Test_ID", "Zone_ID", "Date", "Hour_slot", "Hire_count").filter("Test_ID is not null").orderBy(F.asc("Test_ID"))
final_df.show()


# Write the modified dataframe to csv. 
## Spark write function will split the workload and save the output spread out over multiple parts
## Using cat and >  we will generate a single output file
final_df.write.mode("overwrite").csv('output/attempts/first-attempt')
os.system('rm output/attempts/first-attempt.csv')
os.system('cat output/attempts/first-attempt/p* > output/attempts/first-attempt.csv')

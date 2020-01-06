#!/usr/bin/env python
# coding: utf-8

import pyspark.sql.functions as F

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *


# Create a spark session/application
spark = SparkSession.builder.appName('Homework5_Exercise1').getOrCreate()
sqlcontext = SQLContext(sc)

# Import web-Google.txt as dataframe using the defined schema
schema = StructType(
    [
        StructField("FromNodeId", IntegerType(), False),
        StructField("ToNodeId", IntegerType(), False)
    ]
)
graph_df = spark.read.format("csv").option("header", "true").option("delimiter", "\t").option("comment", "#").schema(schema).load("data/web-Google.txt")

outlinks_df = graph_df.groupBy("FromNodeId").agg(F.countDistinct("ToNodeId").alias("Outlinks")).orderBy(F.desc("Outlinks"))

# Write the output to output/exercise1.csv
outlinks_df.write.mode("overwrite").csv(f'output/exercise1')
os.system(f'rm output/exercise1.csv')
os.system(f'cat output/exercise1/p* > output/exercise1.csv')


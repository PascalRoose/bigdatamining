#!/usr/bin/env python
# coding: utf-8

import numpy
import random
import pyspark.sql.functions as F

from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Create a spark session/application
spark = SparkSession.builder.appName('Homework4_Exercise2').getOrCreate()

# Import ratings.dat
ratings_df = spark.read.format("csv").load("data/ratings.dat")
ratings_df = ratings_df.select(F.split(ratings_df.columns[0], "::").alias('SplitValues'))
ratings_df = ratings_df.withColumn('UserID', F.col('SplitValues').getItem(0).cast(IntegerType()))
ratings_df = ratings_df.withColumn('MovieID', F.col('SplitValues').getItem(1).cast(IntegerType()))
ratings_df = ratings_df.withColumn('Rating', F.col('SplitValues').getItem(2).cast(ByteType()))
ratings_df = ratings_df.withColumn('Timestamp', F.col('SplitValues').getItem(3).cast(IntegerType()))
ratings_df = ratings_df.drop(F.col('SplitValues'))

# Create a matrix with movies as columns and users as rows, each matrix value is a rating
matrix_df = ratings_df.groupby('UserID').pivot('MovieID').sum('Rating').fillna(0)

# Convert matrix to a single column with an array
columns = [c for c in matrix_df.columns if c != 'UserID']
userratings_df = matrix_df.withColumn('Ratings', F.array(columns)).select('UserID', 'Ratings')

# To prove this works for any given user, we pick one at random
userids = ratings_df.select('UserID').distinct().collect()
userid = random.choice(userids)[0]
userratings = userratings_df.select('Ratings').where(F.col('UserID') == userid).collect()[0]['Ratings']

# Calculation the cosine similarity
def cos_sim(other_userratings):
    # The cosine similiarity = dotproduct / ( normalized(a) * normalized(b) )
    dot = numpy.dot(userratings, other_userratings)
    norma = numpy.linalg.norm(userratings)
    normb = numpy.linalg.norm(other_userratings)
    cos = dot/(norma*normb)
    return cos.item()

# Load all userratings into memory
all_userratings = userratings_df.select('UserID', 'Ratings').collect()

# Calculate user simularities
usersim = []
for other_userratings in all_userratings:
    usersim.append((other_userratings['UserID'], cos_sim(other_userratings['Ratings'])))

# Load the user similarities as dataframe to output
schema = StructType(
    [
        StructField("UserID", IntegerType(), True),
        StructField("Rating", FloatType(), True)
    ]
)
output_df = spark.createDataFrame(usersim, schema).sort(F.desc('Rating'))

# Write the output to output/exercise1.csv
output_df.write.mode("overwrite").csv(f'output/exercise2/{userid}')
os.system(f'rm output/exercise2/{userid}.csv')
os.system(f'cat output/exercise2/{userid}/p* > output/exercise2/{userid}.csv')

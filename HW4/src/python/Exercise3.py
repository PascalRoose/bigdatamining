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

# Create a matrix with users as columns and movies as rows, each matrix value is a rating
matrix_df = ratings_df.groupby('MovieID').pivot('UserID').sum('Rating').fillna(0)

# Convert matrix to a single column with an array
columns = [c for c in matrix_df.columns if c != 'MovieID']
movieratings_df = matrix_df.withColumn('Ratings', F.array(columns)).select('MovieID', 'Ratings')

# To prove this works for any given movie, we pick one at random
movieids = ratings_df.select('MovieID').distinct().collect()
movieid = random.choice(movieids)[0]
movieratings = movieratings_df.select('Ratings').where(F.col('MovieID') == movieid).collect()[0]

# Calculation the cosine similarity
def cos_sim(a, b):
    # The cosine similiarity = dotproduct / ( normalized(a) * normalized(b) )
    dot = numpy.dot(a, b)
    norma = numpy.linalg.norm(a)
    normb = numpy.linalg.norm(b)
    cos = dot/(norma*normb)
    return cos.item()

# Load all movieratings into memory
all_movieratings = movieratings_df.select('MovieID', 'Ratings').collect()

# Calculate user simularities
moviesim = []
for other_movieratings in all_movieratings:
    simularity = cos_sim(movieratings['Ratings'], other_movieratings['Ratings'])
    moviesim.append((other_movieratings['MovieID'], simularity))

# Load the movie similarities as dataframe to output
schema = StructType(
    [
        StructField("MovieID", IntegerType(), True),
        StructField("Rating", FloatType(), True)
    ]
)
output_df = spark.createDataFrame(moviesim, schema).sort(F.desc('Rating'))

# Write the output to output/exercise3.csv
output_df.write.mode("overwrite").csv(f'output/exercise3/{movieid}')
os.system(f'rm output/exercise3/{movieid}.csv')
os.system(f'cat output/exercise3/{movieid}/p* > output/exercise3/{movieid}.csv')

#!/usr/bin/env python
# coding: utf-8

import pyspark.sql.functions as F

from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Create a spark session/application
spark = SparkSession.builder.appName('Homework4_Exercise1').getOrCreate()

# Import movies.dat
movies_df = spark.read.format("csv").load("data/movies.dat")
movies_df = movies_df.select(F.split(movies_df.columns[0], "::").alias('SplitValues'))
movies_df = movies_df.withColumn('MovieID', F.col('SplitValues').getItem(0).cast(IntegerType()))
movies_df = movies_df.withColumn('Title', F.col('SplitValues').getItem(1))
movies_df = movies_df.withColumn('Genre', F.split(F.col('SplitValues').getItem(2), '\|'))
movies_df = movies_df.drop(F.col('SplitValues'))

# Import ratings.dat
ratings_df = spark.read.format("csv").load("data/ratings.dat")
ratings_df = ratings_df.select(F.split(ratings_df.columns[0], "::").alias('SplitValues'))
ratings_df = ratings_df.withColumn('UserID', F.col('SplitValues').getItem(0).cast(IntegerType()))
ratings_df = ratings_df.withColumn('MovieID', F.col('SplitValues').getItem(1).cast(IntegerType()))
ratings_df = ratings_df.withColumn('Rating', F.col('SplitValues').getItem(2).cast(FloatType()))
ratings_df = ratings_df.withColumn('Timestamp', F.col('SplitValues').getItem(3).cast(IntegerType()))
ratings_df = ratings_df.drop(F.col('SplitValues'))

'''
# Import users.dat
users_df = spark.read.format("csv").load("data/users.dat")
users_df = users_df.select(F.split(users_df.columns[0], "::").alias('SplitValues'))
users_df = users_df.withColumn('UserID', F.col('SplitValues').getItem(0).cast(IntegerType()))
users_df = users_df.withColumn('Gender', F.col('SplitValues').getItem(1))
users_df = users_df.withColumn('Age', F.col('SplitValues').getItem(2).cast(ByteType()))
users_df = users_df.withColumn('Occupation', F.col('SplitValues').getItem(3).cast(ByteType()))
users_df = users_df.withColumn('Zip-code', F.col('SplitValues').getItem(4).cast(IntegerType()))
users_df = users_df.drop(F.col('SplitValues'))
'''

# Calculate average rating
avgratings_df = ratings_df.select(F.col('MovieID'), F.col('Rating'))
avgratings_df = avgratings_df.groupBy(F.col('MovieID')).agg(F.mean(F.col('Rating')).alias('Average Rating'))
avgratings_df = avgratings_df.join(movies_df, avgratings_df.MovieID == movies_df.MovieID)

# Write the output to output/exercise1.csv
output_df = avgratings_df.select('Title', 'Average Rating').sort(F.desc("Average Rating"))
output_df.write.mode("overwrite").csv('output/exercise1/avgrating')
os.system(f'rm output/exercise1/avgrating.csv')
os.system(f'cat output/exercise1/avgrating/p* > output/exercise1/avgrating.csv')

#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession

from functools import reduce
from operator import add

# Create a spark session/application
spark = SparkSession.builder.appName('Exercise 1').getOrCreate()


# Create a dataframe for each platform with all topics

platforms = ['Facebook', 'GooglePlus', 'LinkedIn']
dataframes = {}

for platform in platforms:
    df = spark.read.format('csv').option('header', 'true').option('delimiter', ',').load(f'data/{platform}_*.csv')
    dataframes[platform] = df


# Generate the dataframes for the average popularity by hour and by day 
#  for each platform for each news article

# Dictionary with each platform as sub-dictionary for storing the generated dataframes
avg_dataframes = {x: {} for x in platforms}

# 3x 20 minutes in an hour, 24x hours,in a day 3*24 = 72
hours = int(144 / 3)
days = int(hours / 24)

# Loop through the platforms and their dataframe
for platform, df in dataframes.items():
    # Calculate the average popularity for each hour 
    for i in range(1, hours + 1):
        # Grab 3 colomns each time calculate the average and add as a new row
        cols = [f'TS{i+j}' for j in range(3)]
        # Calculate the average and add the result as a new colomn
        df = df.withColumn(f'avg_hour{i}', reduce(add, (df[col] for col in cols)) / 3)
    # Create a new dataframe with all of the avg_hour* columns
    avg_dataframes[platform]['avg_hour'] = df.select([f'avg_hour{i}' for i in range(1, hours + 1)])
    
    # Calculate the average popularity for each day 
    for i in range(1, days + 1):
        # Grab 24 colomns each time calculate the average and add as a new row
        cols = [f'avg_hour{i+j}' for j in range(24)]
        # Calculate the average and add the result as a new colomn
        df = df.withColumn(f'avg_day{i}', reduce(add, (df[col] for col in cols)) / 24)
    # Create a new dataframe with all of the avg_day* columns
    avg_dataframes[platform]['avg_day'] = df.select([f'avg_day{i}' for i in range(1, days + 1)])


# For each platform write the avg_hour framework to a txt file as <platform>_hour.txt
#  Each line contains a news article and the average popularity of it per hour comma-seperated

for platform, avg in avg_dataframes.items():
    df = avg['avg_hour']
    cols = [f'avg_hour{i}' for i in range(1, hours + 1)]
    part_path = f'output/exercise2/partitions/{platform}_hour'
    out_path = f'output/exercise2/{platform}_hour'
    df.select(cols).write.mode("overwrite").csv(part_path)
    
    os.system(f'rm {out_path}.txt')
    os.system(f'cat {part_path}/p* > {out_path}.txt')


# For each platform write the avg_hour framework to a txt file as <platform>_hour.txt
#  Each line contains a news article and the average popularity of it per day comma-seperated

for platform, avg in avg_dataframes.items():
    df = avg['avg_hour']
    cols = [f'avg_hour{i}' for i in range(1, days + 1)]
    part_path = f'output/exercise2/partitions/{platform}_day'
    out_path = f'output/exercise2/{platform}_day'
    df.select(cols).write.option("header", True).mode("overwrite").csv(part_path)
    
    os.system(f'rm {out_path}.txt')
    os.system(f'cat {part_path}/p* > {out_path}.txt')

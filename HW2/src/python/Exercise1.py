#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
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

# Make the values of title and headline lowercase 
#  and remove any non alphabetic characters (except spaces)
# Remove hh:mm from publish date
df = df.withColumn('Title', lower(col('Title'))).withColumn('Title', regexp_replace('Title', '[^a-z\s]', '')).withColumn('Headline', lower(col('Headline'))).withColumn('Headline', regexp_replace('Headline', '[^a-z\s]', '')).withColumn("PublishDate", date_format("PublishDate", "yyyy-MM-dd"))

### search values for the 2 cases
cases = ('Headline', 'Title')


# All words and their frequency. Outputs a file for title and headline

def getDf_Total(case):
    ### Counts the words, sorts them from high to low
    df_total = df.withColumn('word', explode_outer(split(col(case), ' '))).groupBy('word').count().sort('count', ascending=False)

    return df_total

def writeDf_Total(df_Total, case):
    ### Writes the output into a file
    ### Sparks partitions the output, we will rectify this later
    df_Total.write.mode("overwrite").option("delimiter", " ").csv(f'output/exercise1/total/{case}/partitions/Exercise1_{case}_Total')

    ### Remove old instance of output file
    os.system(f'rm output/exercise1/total/{case}/partitions/Exercise1_{case}_Total.txt')
    ### Concat al the partitions into an txt file
    os.system(f'cat output/exercise1/total/{case}/partitions/Exercise1_{case}_Total/p* > output/exercise1/total/Exercise1_{case}_Total.txt')


for case in cases:
    ### Get dataframe with the words counted and sorted
    df_temp = getDf_Total(case)
    ### Write the output to a file
    writeDf_Total(df_temp, case)


# All words per day (in title and headline) and their frequency. 
# Outputs a file for title and headline for every day

def processDf_perDay(df_total, case):
    ### Makes a list of dataframes
    ### Each new dataframes contains the articles published on the same day
    
    ### Collects unique dates
    unique_dates = df.agg(collect_set("PublishDate")).collect()[0][0]
    
    ### List to store dataframes for articles published on the same day
    df_dates_temp = {}

    for date in unique_dates:
        df_dates_temp[date] = df_total.filter(col('PublishDate') == date)
    
    df_dates = {}
    
    for date, df_date in df_dates_temp.items():
        df_dates[date] = df_date.withColumn("word", explode(split(col(case), " "))).groupBy("word").count().sort("count", ascending=False)
    
    return df_dates

def writeDF_PerDay(df_date, date, case):
    ### Write the output to a file
    ### Sparks partitions the output, we will rectify this later
    df_date.write.mode("overwrite").option("delimiter", " ").csv(f'output/exercise1/perDay/partitions/Exercise1_{case}_{date}')

    ### Remove old instance of output file
    os.system(f'rm output/exercise1/perDay/Exercise1_{case}_{date}.txt')
    ### Concat al the partitions into an txt file
    os.system(f'cat output/exercise1/perDay/partitions/Exercise1_{case}_{date}/p* > output/exercise1/perDay/Exercise1_{case}_{date}.txt')


for case in cases:
    ### Count the words in each dataframe
    df_dates = processDf_perDay(df, case)
    for date, df_date in df_dates.items():
        ### Write the dataframe to a file
        writeDF_PerDay(df_date, date, case)


# All words per topic and their frequency. Outputs a file for every topic

def getDf_perTopic(topic, case):
    ### Collects all the articles for the topic
    df_temp_uncounted = df.filter(col('Topic') == topic)
    
    ### Counts, removes unwanted words ('...') & sorts from high to low
    df_temp_counted = df_temp_uncounted.withColumn('word', explode(split(col(case), ' '))).groupBy('word').count().sort('count', ascending=False)
    return df_temp_counted

def write_perTopic(df_temp_counted, topic, case):
    ### Write the output to a file
    ### Sparks partitions the output, we will rectify this later
    df_temp_counted.write.mode("overwrite").option("delimiter", " ").csv('output/exercise1/perTopic/'+case+'/partitions/Exercise1_'+case+'_perTopic_'+topic)
    
    ### Remove old instance of output file
    os.system('rm output/exercise1/perTopic/'+case+'/partitions/Exercise1_'+case+'_perTopic_'+topic+'.txt')
    ### Concat al the partitions into an txt file
    os.system('cat output/exercise1/perTopic/'+case+'/partitions/Exercise1_'+case+'_perTopic_'+topic+'/p* > output/exercise1/perTopic/'+case+'/Exercise1_'+case+'_perTopic_'+topic+'.txt')

### Search values for the 4 topics
### In the first place we wanted to look up the topics dynamically
### But this gave way more topics then the 4 topics needed for the assignment
### We therefore only search for articles with 1 of the 4 topics asked in the assignment
topics = ('obama', 'economy', 'palestine', 'microsoft')

### For both of the 2 cases
for case in cases:
    ### For all of the 4 topics
    for topic in topics:
        ### collect the topics, count and write
        df_temp_counted = getDf_perTopic(topic, case)
        write_perTopic(df_temp_counted, topic, case)

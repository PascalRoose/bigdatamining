#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import split, explode, monotonically_increasing_id, col, lower, regexp_replace

# Create a spark session/application
spark = SparkSession.builder.appName('Exercise 4').getOrCreate()

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

# Mutate the dataframe so we only have the colomns we need in the right format
# Title and headline will be lowercase and only include alphabetic characters
df = df.withColumn('Title', lower(col('Title'))).withColumn('Title', regexp_replace('Title', '[^a-z\s]', '')).withColumn('Headline', lower(col('Headline'))).withColumn('Headline', regexp_replace('Headline', '[^a-z\s]', '')).select('Title', 'Headline', 'Topic')

### search values for the 2 cases
cases = ('Headline', 'Title')
topics = ('obama', 'economy', 'palestine', 'microsoft')

def get_top100(case, topic):
    ### Collects all the articles for the topic
    df_temp_untop100ed = df.filter("Topic == '"+topic+"'")
    
    ### Explode title or headline, count each word, sort descending and take the first 100 results
    df_temp_top100ed = df_temp_untop100ed.withColumn('word', explode(split(col(case), ' '))).groupBy('word').count().sort('count', ascending=False).limit(100)
    
    # Convert the dataframe to a list
    top100 = [row[0] for row in df_temp_top100ed.collect()]
    
    return top100

# Create a dictionary with top100 list for each topic per case
top100_dict = {}
for case in cases:
    top100_dict[case] = {}
    for topic in topics:
        top100_dict[case][topic] = get_top100(case, topic)

def create_comatrix(case, topic, top100):
    # Filter by topic, explode the case (either title or headline)
    df_expl = df.filter("Topic == '"+topic+"'").withColumn("id", monotonically_increasing_id()).select("id", explode(split(case, " ")))
    # Filter all of the words, leave only the words that are in the top100
    df_fltr = df_expl.filter(df_expl.col.isin(top100))
    
    # Use join and crosstab to calculate and create a co-occurence matrix
    return df_fltr.withColumnRenamed("col", "col_").join(df_fltr, ["id"]).stat.crosstab("col_", "col")

# Create a dictionary with coocurrence df for each topic per case
comatrix_dict = {}
for case in cases:
    comatrix_dict[case] = {}
    for topic in topics:
        top100 = top100_dict[case][topic]
        comatrix_dict[case][topic] = create_comatrix(case, topic, top100)

# Write all of the files
for case in cases:
    for topic in topics:
        comatrix_df = comatrix_dict[case][topic]
        
        # Grab only the first line with header and write
        comatrix_df.limit(1).write.option("header", True).mode("overwrite").csv(f"output/exercise4/partition/comatrix_{case}_{topic}_header")
        comatrix_df.write.mode("overwrite").csv(f"output/exercise4/partition/comatrix_{case}_{topic}")
        
        # Remove the old file
        os.system(f'rm output/exercise4/comatrix_{case}_{topic}.txt')
        
        # Write the header first (first line of _header with head -1)
        os.system(f'head -1 output/exercise4/partition/comatrix_{case}_{topic}_header/p* > output/exercise4/comatrix_{case}_{topic}.txt')
        
        # Write all of the partitions to one txt file
        os.system(f'cat output/exercise4/partition/comatrix_{case}_{topic}/p* >> output/exercise4/comatrix_{case}_{topic}.txt')

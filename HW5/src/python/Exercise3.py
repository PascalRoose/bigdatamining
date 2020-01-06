#!/usr/bin/env python
# coding: utf-8

import random
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

# Get a dictionary of inlinks and outlinks both as lists
def node_connectivity(node_v):
    out_links = graph_df.filter(F.col('FromNodeId') == node_v)
    out_links = [str(out_link['ToNodeId']) for out_link in out_links.collect()]
    
    in_links = graph_df.filter(F.col('ToNodeId') == node_v)
    in_links = [str(in_link['FromNodeId']) for in_link in in_links.collect()]
    
    return {"out_links": out_links, "in_links": in_links}

# Get a list of all nodes
from_nodes = [row["FromNodeId"] for row in graph_df.select("FromNodeId").distinct().collect()]
to_nodes = [row["ToNodeId"] for row in graph_df.select("ToNodeId").distinct().collect()]
nodes = list(set(from_nodes + to_nodes))

# Select a random node v
node_v = random.choice(nodes)

connectivity_v = node_connectivity(node_v)

# Write the output to output/exercise3/{node_v}.csv
with open(f'output/exercise3/{node_v}.txt', 'w+') as output_file:
    output_file.write(",".join(connectivity_v['out_links']) + '\n')
    output_file.write(",".join(connectivity_v['in_links']))

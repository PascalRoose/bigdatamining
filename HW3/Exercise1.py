#!/usr/bin/env python
# coding: utf-8

import binascii
import itertools
import os

from bs4 import BeautifulSoup
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, countDistinct

# Create a spark session/application
spark = SparkSession.builder.appName('Exercise3_1').getOrCreate()
spark.conf.set('spark.sql.pivotMaxValues', 22000)


# Get all of the articles as a list of words
## NOTE: Some articles appear to not have a body, therefore having an empty list of words

articles = []

#Loop through all files in data/ (segments of the dataset)
files = os.listdir('data')
for file in files:
    with open(f'data/{file}', 'rb') as data:
        soup = BeautifulSoup(data)
    # An article is defined by <reuters> tag
    seg_articles = soup.find_all('reuters')
    for seg_article in seg_articles:
        # The text of the article in <body> tag
        text = seg_article.find('body')
        if text is not None:
            # Split the text into words, decapitalize all words, remove all dots and commas
            article_words = text.next_element.split()
            article_words = [word.lower().replace(',', '').replace('.', '') for word in article_words]
        else:
            # If the <body> tag was not found we assume the article has no words and therefore is an empty list
            article_words = []
        articles.append(article_words)


# Define all documents as a set of k-shingles, where k = 3 (words)

docs = []
for doc_id, words in enumerate(articles):
    doc = []
    # Make sure the article has at least k words
    if len(words) >= 3:
        for index in range(0, len(words) - 2):
            shingle = words[index]                + " " + words[index+1]                + " " + words[index+2]
            # Hash the shingle to a 32bit integer
            hash_shingle = binascii.crc32(shingle.encode()) & 0xffffffff
            doc.append(hash_shingle)
    # Make sure the document is a set of k-shingles, so no duplicates. Convert back to a list.
    docs.append([doc_id, list(set(doc))])


# Create a set representation as MxN where N=docs, M=shingles
## NOTE: We did not generate an output file for this. 
##  The output would have been about 3.4GB, way too big and sparse to even read anyway 
##  and requires too much memory memory to generate

df = spark.createDataFrame(docs, ['doc', 'shingles'])
exploded_df = df.select(df.doc, explode(df.shingles).alias('shingle'))
# Group by shingles, set shingles as rows (pivot), set the value as sum of the occurences per document (1)
#  fill the empty cells with 0, drop the shingles column
matrix_df = exploded_df.groupby('shingle').pivot('doc').sum('doc').fillna(0).drop('shingle')

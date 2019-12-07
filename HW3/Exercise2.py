#!/usr/bin/env python
# coding: utf-8

import csv
import binascii
import itertools
import os

import pyspark.sql.functions as F

from bs4 import BeautifulSoup
from pyspark.sql import SparkSession
from datasketch import MinHash
from datasketch.hashfunc import sha1_hash32
from random import randrange

# Create a spark session/application
spark = SparkSession.builder.appName('Exercise3_2').getOrCreate()
spark.conf.set('spark.sql.pivotMaxValues', 22000)


# Generate k-shingles for all documents. See Exercise 1 for more explaination and comments

articles = []
files = os.listdir('data')
for file in files:
    with open(f'data/{file}', 'rb') as data:
        soup = BeautifulSoup(data)
    seg_articles = soup.find_all('reuters')
    for seg_article in seg_articles:
        text = seg_article.find('body')
        if text is not None:
            article_words = text.next_element.split()
            article_words = [word.lower().replace(',', '').replace('.', '') for word in article_words]
        else:
            article_words = []
        articles.append(article_words)

docs = []
for id, article_words in enumerate(articles):
    doc = []
    if len(article_words) >= 3:
        for index in range(0, len(article_words) - 2):
            shingle = article_words[index]                + " " + article_words[index+1]                + " " + article_words[index+2]
            hash_shingle = binascii.crc32(shingle.encode()) & 0xffffffff
            doc.append(hash_shingle)
    docs.append([id, list(set(doc))])

# We hashed the documents to k-shingles as 32bit integers
# This is the first prime after 2^32 - 1
prime_number = 4294967311

# Biggest possible hash value (2^32 - 1).
max_hash = (1 << 32) - 1

# Amount of permutations, or the amount of hash functions used.
permutations = 256

# List of x tuples of (a,b) where x is the number of permutations.
coefficients = []
for i in range(permutations):
    a = randrange(0, max_hash)
    b = randrange(0, max_hash)
    coefficients.append((a,b))

def hash_function(a, b, x):
    hx = (a * x + b) % prime_number
    return hx


# Generate the 256 minhash signatures for all documents

minhash_sigs = []
for hashfunc in range(no_hashfuncs):
    minhash_sig = []
    for doc in docs:
        hash_sigs = []
        # Document is a tuple of (doc_id, [shingles]). We want the list of shingles so doc[1]
        for shingle in doc[1]:
            a,b = coefficients[hashfunc]
            hash_sig = hash_function(a, b, shingle)
            hash_sigs.append(hash_sig)
        # If the list of hash signatures is empty the document is empty,
        #  so give it the minhash signature of 0
        if len(hash_sigs) == 0:
            minhash = 0
        else:
            minhash = min(hash_sigs) 
        minhash_sig.append(minhash)
    minhash_sigs.append(minhash_sig)


# Write the minhash signatures to output/exercise2.csv
# The output is a HxM matrix where H=256, M=21578
with open("output/exercise2.csv", "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerows(minhash_sigs)

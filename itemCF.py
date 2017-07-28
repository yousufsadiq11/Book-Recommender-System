# Item-Item Collaborative Filtering on Books DataSet using pySpark with cosine similarity and weighted sums
import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import random
from pyspark import SparkContext
from pyspark import SparkConf

def calculate_Cosine_Similarity_and_size(pairs_of_items,pairs_of_ratings):
    #Extracting Size and Similarity Measure
    aa=0.0
    a_plus_b=0.0
    bb=0.0
    size = 0.0
    for each_pair in pairs_of_ratings:
        bb += np.float(each_pair[1]) * np.float(each_pair[1])
        aa += np.float(each_pair[0]) * np.float(each_pair[0])
        a_plus_b += np.float(each_pair[0]) * np.float(each_pair[1])
        size += 1
    product_of_square_roots=np.sqrt(aa)*np.sqrt(bb)
    dot_product=a_plus_b
    #Computing Cosine Value from the Vectors
    if product_of_square_roots:
        cosine_similarity_measure = dot_product/float(product_of_square_roots)
    else:
        cosine_similarity_measure=0.0
    return pairs_of_items, (cosine_similarity_measure,size)

def pickingRandomRatings(user_id,items,n):
    #Picking few random ratings from the list when the list size exceeds a predefined value n
    if len(items) > n:
        return user_id, random.sample(items,n)
    else:
        return user_id, items

def pickingTop_50_recommendations(user_id,items_with_ratings,similarity_values,n):
    #Picking top N values for each user
    #Using dictionary for storing measure of each individual item to get values from all regions
    totals = defaultdict(int)
    similarity_summation = defaultdict(int)
    for (items,rating) in items_with_ratings:
        # Searching for nearest neighbour corresponding to this item
        neighbors = similarity_values.get(items,None)
        if neighbors:
            for (neighbor,(sim,count)) in neighbors:
                if neighbor != items:
                    totals[neighbor] += sim * rating
                    similarity_summation[neighbor] += sim
    scored_items = [(total/similarity_summation[item],item) for item,total in totals.items()]
    # Sorting
    scored_items.sort(reverse=True)
    return user_id,scored_items[:n]

def formingKeys(pairs_of_items,value):
    #Making first item as key to avoid repeating pairs
    (item1,item2) = pairs_of_items
    return item1,(item2,value)

def extractInputFileData(input_data_line):
    #extracting data from input file by using the comma as seperator
    input_data_line = input_data_line.split(",")
    return input_data_line[0],(input_data_line[1],float(input_data_line[2]))

def bookNameExtraction(splitted_data):
	splitted_data = splitted_data.split(",")
	return splitted_data[0],splitted_data[1]

def formingItemPairs(user_id,items_and_their_ratings):
    #Retrieving item pairs for each users
    return [[(first_item[0],second_item[0]), (first_item[1],second_item[1])] for (first_item, second_item) in combinations(items_and_their_ratings,2)]

def pickingNearestNeighbours(key,similarity_values,n):
    #Picking top n nearest values
    similarity_values = list(similarity_values)
    similarity_values.sort(key=lambda x: x[1][0],reverse=True)
    return key, similarity_values
sc.stop()
sc = SparkContext(appName="SBG")
lines = sc.textFile(argv[0])
book_names = argv[1]
#obtaining user followed by item and ratings pairs
print("Running")
users_item_pairs = lines.map(extractInputFileData).groupByKey().map(
lambda x: pickingRandomRatings(x[0],x[1],500)).cache()
#Finding item pairs and their ratings for Cosine Calculation
item_pairs = users_item_pairs.filter(
lambda z: len(z[1]) > 1).map(
lambda y: formingItemPairs(y[0],y[1])).flatMap(lambda x:x).groupByKey()
#print(item_pairs.mapValues(list).collect())
#Computing Cosine Similarity for each pair followed by the size
#Making first item as key for easy retrieval
similarity_computation = item_pairs.map(
lambda x: calculate_Cosine_Similarity_and_size(x[0],x[1])).map(
lambda x: formingKeys(x[0],x[1])).groupByKey().map(
lambda x: pickingNearestNeighbours(x[0],x[1],50)).collect()
#Putting it in a dictionary
item_sim_dict = {}
for (item,data) in similarity_computation:
    item_sim_dict[item] = data
    isb = sc.broadcast(item_sim_dict)
#Picking to 50 Values
user_item_recs = users_item_pairs.map(
lambda p: pickingTop_50_recommendations(p[0],p[1],isb.value,50))
#The next commented line can be used to print recommendations for all users
#print(user_item_recs.collect())
#User for which recommendation has to be suggested
user_id = '8'
#Retrieving the suggestions for the corresponding user requested
user_suggestion_list = sc.parallelize(user_item_recs.filter(lambda x:x[0]==user_id).values().collect()[0])
book_id_for_suggestions = user_suggestion_list.map(lambda x:x[1]).collect()
print(book_id_for_suggestions)
book_name_path = sc.textFile(book_names)
Retrieving_Book_Names = book_name_path.map(lambda x:x.encode("ascii","ignore")).map(bookNameExtraction).filter(lambda x:x[0] in book_id_for_suggestions).map(lambda x:x[1]).collect()
print(Retrieving_Book_Names)
#Output path where recommendations for a particular user are stored
with open("outputpath", "w") as o:
	o.write("\n".join([str(x) for x in Retrieving_Book_Names]))
sc.stop()
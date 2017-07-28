from pyspark import SparkContext
import sys
from math import sqrt

compare_user_count = 10
Similarity_Evaluation = "pearson_correlation"

# Generates Vector for user
def Similarity_Vector_For_User(ratings, compared_users):
    global Similarity_Evaluation
    return sorted(map(lambda compared_user: [compared_user[0], similarity_function(ratings, compared_user[1])], compared_users), key = lambda x: -x[1])

# Predicts ratings of user based on other heavy weighted users
def estimate_Rating(item, compared_users, similarity_vectors):
  normalizing_value = 0
  computation_value = 0.0
  # Looping for every heavy weight each_user to be compared
  for (each_user, ratings) in compared_users:
    for (itemid, rating) in ratings:
        # Check if it matches with the item id and proceeding with computation
      if itemid == item:
        computation_value += similarity_vectors[each_user] * rating
        normalizing_value += 1

  return computation_value / max(normalizing_value, 1)

# Pearson Correlation for Similarity Computation
def Pearson_Correlation(rating_1, rating_2):
  computation_value = 0.0
  item1 = set([x[0] for x in rating_1])
  item2 = set([x[0] for x in rating_2])
  avg_rating_user1 = sum(map(lambda x: x[1], rating_1)) / len(rating_1)
  average_rating_user2 = sum(map(lambda x: x[1], rating_1)) / len(rating_2)
  # Mapping to Dictionary for Swift Retrieval
  map_rating_1 = dict(rating_1)
  map_rating_2 = dict(rating_2)
  # Finding Common Items for Similarity Computation and proceeding if there are similar items
  similar_items = item1.intersection(item2)
  if similar_items:
    for each_similar_item in similar_items:
      rating_by_1 = map_rating_1.get(each_similar_item, 0)
      rating_by_2 = map_rating_2.get(each_similar_item, 0)
      # Subtracting mean value from users
      diff1 = rating_by_1 - avg_rating_user1
      diff2 = rating_by_2 - average_rating_user2
      computation_value += (diff1 * diff2) / max(sqrt(diff1 * diff1 + diff2 * diff2), 1.0)

  return computation_value

if 7 == 7:
  sc.stop()
  # Spark Context
  sc=SparkContext(appName="User Based Recommender System")
  # User Ratings File Path
  user_ratings_file = sys.argv[1]
  # Book Titles File Path
  Book_Titles = sys.argv[2]
  Similarity_Evaluation = Pearson_Correlation
  # Count of number of users against whom it has to be compared
  compare_user_count = 20
  # Count of Number of recommendations to be provided to user
  count_of_recommendations = 8
  # User ID of user to whom recommendations has to be provided
  user_id = sys.argv[3]
  # Extracting Book Titles and Data from Ratings File and Book titles file
  Book_Titles_Extraction = sc.textFile(Book_Titles).map(lambda line: line.strip().split(",")).map(lambda x: (int(x[0]), x[1])).collectAsMap()
  Ratings_File_Extraction = sc.textFile(user_ratings_file).map(lambda line: line.strip().split(",")).map(lambda x: (int(x[0]), (int(x[1]), float(x[2]))))
  # Obtaining the ID of all ISBN
  ISBN = Ratings_File_Extraction.map(lambda x: x[1][0]).distinct()
  # Formation of vectors for users
  all_users = Ratings_File_Extraction.groupByKey().map(lambda x: (x[0], x[1], sqrt(sum(map(lambda rating: rating[1] * rating[1], x[1]))))).map(lambda x: (x[0], map(lambda rating: (rating[0], rating[1] / (0.000001 + x[2])), x[1])))
  # Picking the top users against whom similarity measure has to be compared
  compared_users = all_users.takeOrdered(compare_user_count, key=lambda x: -len(x[1]))
  # Picking out the current user based on user_id provided as input
  input_user  = all_users.filter(lambda x: x[0] == user_id).first()
  # Input User Ratings
  user_items = [item[0] for item in input_user[1]]
  similarity_vector = dict(Similarity_Vector_For_User(input_user[1], compared_users))
  Unrated_items = ISBN.filter(lambda item: item not in user_items)
  # Current User Recommendations List
  current_user_recommendations = Unrated_items.map(lambda item: (item, estimate_Rating(item, compared_users, similarity_vector))).takeOrdered(count_of_recommendations, lambda x: -x[1])
  print(current_user_recommendations)
  sc.stop()
  # Output File Path where Output has to be writted and stored
  with open("C:\\Users\\sadiq\\Desktop\\test\\output_user.txt", "w") as o:
    o.write("\n".join(map(lambda x: Book_Titles_Extraction.get(x[0], "books"), current_user_recommendations)))
  sc.stop()
import sys
import timeit;
from pyspark import SparkContext
import numpy as np
from numpy.random import rand
from numpy import matrix
import pandas as pd
    
   
def calculateUser(x):  
    user = users_brod.value.T * users_brod.value
    for a in range(factor):
        user[a, a] = user[a,a] + lambdaVal * n
    new_user = users_brod.value.T * user_rating_brod.value[x,:].T
    return np.linalg.solve(user, new_user)

def calculatebook(x):
    userRat = user_rating_brod.value.T
    books = books_brod.value.T * books_brod.value
    for b in range(factor):
        books[b, b] = books[b,b] + lambdaVal * m
        
    new_book = books_brod.value.T * userRat[x, :].T
    return np.linalg.solve(books, new_book)

# function to calulate Root mean square error        
def rmse(user_rating, books, users):
    bookUser = books * users.T
    diff = user_rating - bookUser
    s_diff = (np.power(diff, 2)) / (books_row * users_row)
    return np.sqrt(np.sum(s_diff))
	
if __name__ == "__main__":
    #sc.stop()
    sc = SparkContext(appName="ALS")
    
	# Initializing the parameters: iteration, lambdaVal and factor
    lambdaVal = 0.001
    iteration =  10
    factor = 10
    rms = np.zeros(iteration)
    start = timeit.default_timer()
   
    #ratingsHeaders = ['ISBN', 'Book_Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
    ratingsHeaders = ['ISBN', 'Book_Title']
    ratings = pd.read_table(sys.argv[2], sep=',', header=None, names=ratingsHeaders, skiprows=1)
    bookTitles = ratings.Book_Title.tolist()

    df = pd.read_csv(sys.argv[1], sep=',', header=None, skiprows=1)
    df = df.pivot_table(2,0,1).fillna(0)
    user_rating1 = np.matrix(df.as_matrix())
    
    m,n = user_rating1.shape
    user_rating_brod = sc.broadcast(user_rating1)
	
# Initializing weights matrix using user_rating matrix
    w = np.zeros(shape=(m,n))

    for r in range(m):
        for j in range(n):
            if user_rating1[r,j]>0.5:
                w[r,j] = 1.0
            else:
                w[r,j] = 0.0
    w = w.astype(np.float64, copy=False)
      
# Randomly initialize books and users matrix
    books = 10 * matrix(rand(m, factor))
    books_brod = sc.broadcast(books)
    
    users = 10 * matrix(rand(n, factor))
    users_brod = sc.broadcast(users)
    
    books_row,books_col = books.shape
    users_row,users_col = users.shape

    i = 0

    while (i<iteration):
# solving for books matrix by keeping user matrix constant
        books = sc.parallelize(range(books_row)).map(calculateUser).collect()
        books_brod = sc.broadcast(matrix(np.array(books)[:, :]))

# solving for user matrix by keeping books matrix constant 
        users = sc.parallelize(range(users_row)).map(calculatebook).collect()
        users_brod = sc.broadcast(matrix(np.array(users)[:, :]))

        error = rmse(user_rating1, matrix(np.array(books)), matrix(np.array(users)))
        rms[i] = error
        i = i+1
    
    fi_user = np.array(users).squeeze()
    fi_book = np.array(books).squeeze()
    final = np.dot(fi_book,fi_user.T)
    
    final -= np.min(final)
    final *= float(10) / np.max(final)
    rec_book = np.argmax(final * w,axis =1)
    
# Predicting books for each users
    for u in range(books_row):
        r = rec_book.item(u)
        p = final.item(u,r)
        print ('Prediction for user %d: book_Name %s' %(u+1,bookTitles[r+1]) )
        
    print "RMSE after each iteration: ",rms
    
    stop = timeit.default_timer()
    print "Time: ",stop - start
    print "Avg RMSE:",np.mean(rms)
    sc.stop()

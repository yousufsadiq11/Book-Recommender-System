#Book Recommendation System using Spark

Report Link:
https://sites.google.com/uncc.edu/ymohamme/home

Book Recommendation using Spark on a real - time dataset called Book Crossings which recommends top ten books to the user using Collaborative filtering based on Memory based and Model based filtering approach.

Technologies / Frameworks used: Spark, Python
Tools: Anaconda Jupyter, Cloudera

User.py: This file includes the source code for User based recommendation system implemented through Spark framework.

ItemCF.py: This file includes the source code for Item based recommendation system implemented through Spark framework.

ALS.py: This file is used to provide recommendations to the user based on ALS algorithm implemented through Spark framework.

Below steps are for executing the code in dsba-cluster. Using below steps we can run User-User based collaborative filtering, Item-Item based collaborative filtering and Alternate least square error

Step 1: Copy the code file and input files into dsba-cluster
		
		pscp <code file> <username>@dsba-hadoop.uncc.edu:/users/<username>/.
		
		pscp <input file> <username>@dsba-hadoop.uncc.edu:/users/<username>/.
		
		Eg:	pscp ALS.py <username>@dsba-hadoop.uncc.edu:/users/<username>/.

Step 2: Login to dsba cluster
		
		ssh -X <user_name>@dsba-hadoop.uncc.edu

Step 3: Copy the input files into hadoop filesystem
		
		hadoop fs -put <input file> /user/<username>/
		
		eg: hadoop fs -put Books-Ratings.csv /user/<username>/

Step 4: Execute the code using spark-submit and at the same time copy the output to some file
		
		------------------------------------------------------------
		
		To execute ALS
		
		Eg: spark-submit <code file> <input file with ratings> <input file wiith titles> > output.out
		
		Sometimes if you are running a huge input file you might have to give optional commands for spark-submit as below
		
		spark-submit --executor-memory 13G --num-executors 50 --conf "spark.kryoserializer.buffer.max=1024m" <code file> <input file with ratings> <input file with titles> > output.out
		
		Eg: spark-submit --executor-memory 13G --num-executors 50 --conf "spark.kryoserializer.buffer.max=1024m" ALS.py BX-Book-Ratings-Final.csv BX-Books-Final.csv > output.out
		
		-------------------------------------------------------------
		
		
		To execute Item-Item collaborative filtering
		
		Eg: spark-submit <code file> <input file with ratings> <input file wiith titles> <user-id> > output.out
		
		Sometimes if you are running a huge input file you might have to give optional commands for spark-submit as below
		
		spark-submit --executor-memory 13G --num-executors 50 --conf "spark.kryoserializer.buffer.max=1024m" <code file> <input file with ratings> <input file with titles> <user-id> > output.out
		
		Eg: spark-submit --executor-memory 13G --num-executors 50 --conf "spark.kryoserializer.buffer.max=1024m" ALS.py BX-Book-Ratings-Final.csv BX-Books-Final.csv 10 > output.out
		
		-------------------------------------------------------------

		To execute User-User collaborative filtering
		
		Eg: spark-submit <code file> <input file with ratings> <input file wiith titles> <user-id> > output.out
		
		Sometimes if you are running a huge input file you might have to give optional commands for spark-submit as below
		
		spark-submit --executor-memory 13G --num-executors 50 --conf "spark.kryoserializer.buffer.max=1024m" <code file> <input file with ratings> <input file with titles> <user-id> > output.out
		
		Eg: spark-submit --executor-memory 13G --num-executors 50 --conf "spark.kryoserializer.buffer.max=1024m" ALS.py BX-Book-Ratings-Final.csv BX-Books-Final.csv 10 > output.out
		
		-------------------------------------------------------------

Step 5: Check the output values
		
		Eg: cat output.out

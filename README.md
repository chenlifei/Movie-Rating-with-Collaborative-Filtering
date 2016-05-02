# Movie-Rating-with-Collaborative-Filtering
Movie Rating with Collaborative Filtering in Local (Python) and Spark (Scala)

The project has codes both in python and scala.

Python:
There are two version of codes in python, one is naive version and the other one uses numpy library.
Iteration times is 10 and RMSE would be calculated and output each time.
uSmall.data have 1000 lines data as small data set test.
ml-20m/ratings.csv have 20m lines data as full data set.  (download link: http://grouplens.org/datasets/movielens/)

Spark:
Assume that you have installed the spark in the machine.
Under spark environment (or have set the scala/spark path):
Use “scalac MovieRecommendation.scala” to create jar file.
and then use “spark-submit <jar_name>.jar” to run.

If you cannot run by the above steps, try the following:
“spark-shell” to enter the spark shell.
In spark shell:
“:load MovieRecommendation.scala” (“MovieRecommendation.scala” must be in the same directory of spark-shell)
Copy the code from line 28 to line 79, then paste it to the shell directory


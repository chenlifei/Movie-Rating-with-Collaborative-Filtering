/*
 * Create by Xinghao Zhou
 * A spark program of movie recommendation
 * Team project of "How To Write Fast Code"
 * 2016 April
 * Team Member: Jingwen Qiang, Lifei Chen, Xinghao Zhou
 */

import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.{ PairRDDFunctions, RDD }
import org.apache.spark.SparkContext
import scala.collection.mutable.HashMap
import java.util.List
import java.util.ArrayList
import java.util.Date;

object MovieRecommendation {
  def main(args: Array[String]) {
    var conf = new SparkConf().setAppName("Moive Recommendation")
                          //.setSparkHome("D:\\work\\hadoop_lib\\spark-1.1.0-bin-hadoop2.4\\spark-1.1.0-bin-hadoop2.4")
                          //.setSparkHome("~/")
  
    val sc = new SparkContext(conf);
    // val data = sc.textFile("train")
    
    val beforeDate = new Date()
    val beforeTime = beforeDate.getTime()
    
    val data = sc.textFile("/input/ratings.csv")
    
    val ratings = data.map(_.split(",") match {
      case Array(user, item, rate, time) => Rating(user.toInt, item.toInt, rate.toDouble)
    })
    
    // Get the ALS model.
    val model = new ALS()
      model.setRank(10)
      model.setIterations(10)
      model.setLambda(0.01)
    //.setImplicitPrefs(true)
      model.setUserBlocks(8)
      model.setProductBlocks(8)
      
    val matrix = model.run(ratings)
     
    /*
     * Calculate the MSE
     */
     // Train the model
    val usersProducets = ratings.map(r => r match {
      case Rating(user, product, rate) => (user, product)
    })

    // Predict
    val predictions = matrix.predict(usersProducets).map(u => u match {
      case Rating(user, product, rate) => ((user, product), rate)
    })

    // Coalesce
    val ratesAndPreds = ratings.map(r => r match {
      case Rating(user, product, rate) =>
        ((user, product), rate)
    }).join(predictions)

    // Calculate the MSE
    val MSE = ratesAndPreds.map(r => r match {
      case ((user, product), (r1, r2)) =>
        var err = (r1 - r2)
        err * err
    }).mean()

    println("Mean Squared Error = " + MSE)
    val afterDate = new Date()
    val afterTime = afterDate.getTime()
    // Get the total runtime.
    val totalTime = afterTime - beforeTime
    println("Total time = " + totalTime)
    
   
  }
  
}
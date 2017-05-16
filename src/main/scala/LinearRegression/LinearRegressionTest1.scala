package LinearRegression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by cluster on 2017/4/11.
  */
object LinearRegressionTest1 {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf().setMaster("local[4]").setAppName("line")
    val sc = new SparkContext(conf)

    val data = sc.textFile("lpsa")
    val parsedData = data.map(line =>{
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    })

    val numIterations = 100
    val model = LinearRegressionWithSGD.train(parsedData,numIterations)

    val valuesAndPreds = parsedData.map(point =>{
      val prediction = model.predict(point.features)
      (point.label,prediction)
    })

    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2)}.reduce (_ + _) / valuesAndPreds.count

    println("training Mean Squared Error = " + MSE)
    sc.stop()

  }
}

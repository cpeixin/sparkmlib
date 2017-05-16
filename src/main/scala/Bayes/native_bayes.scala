package Bayes

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by cluster on 2017/4/18.
  */
object native_bayes {
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  def main(args: Array[String]) {
    //1.构建spark对象
    val conf = new SparkConf().setAppName("bayes").setMaster("local[2]")
    val sc = new SparkContext(conf)

    //2.读取样本数据，转换成LablePoint格式
    val data = sc.textFile("data/sample_naive_bayes_data.txt")
    val parsedData = data.map(line => {
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    })

    //3.样本数据划分训练样本和测试样本
    val splitData = parsedData.randomSplit(Array(0.6,0.4),seed = 11L)
    val training = splitData(0)
    val test = splitData(1)

    //4.新建贝叶斯模型  并且训练
    val model = NaiveBayes.train(training,lambda = 1.0,"multinomial")

    //5.对样本进行测试

    val predictionAndLabel = test.map(p => (model.predict(p.features),p.label))

    val print_predict = predictionAndLabel.take(20)

    println("prediction   "+"\t"+"  label")
    for (i <- 0 to print_predict.length - 1){
      println(print_predict(i)._1+"               "+print_predict(i)._2)
    }

  }
}

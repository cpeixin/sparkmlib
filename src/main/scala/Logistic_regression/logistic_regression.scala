package Logistic_regression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object logistic_regression {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("logistic_regression").setMaster("local[2]")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)
    //读取样本数据
    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt")

//  将样本数据分为训练样本数据和测试样本数据 该函数根据weights权重，将一个RDD切分成多个RDD。
//  该权重参数为一个Double数组  每个参数相加得 1
//  第二个参数为random的种子，基本可忽略。
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)


    //新建逻辑回归模型，并且训练。setNumClasses目标列类别个数
    val model = new LogisticRegressionWithLBFGS().
      setNumClasses(10).
      run(training)

    //对测试样本进行测试
    val predictionAndLabels = test.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    val print_predict = predictionAndLabels.take(20)
    println("prediction" + "\t" + "label")
    for (i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }

    //误差计算
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision = " + precision)


//    val ModelPath = "/user/huangmeiling/logistic_regression_model"
//    model.save(sc, ModelPath)
//    val sameModel = LogisticRegressionModel.load(sc, ModelPath)

  }

}

package Bayes

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{SQLContext, Row}

/**
  * Created by cluster on 2017/4/19.
  */
object native_bayes_CN {
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  case class RowBayes(category: String,text: String)
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("bayes").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    val srcRDD = sc.textFile("data/sougou/*.txt").map( line=>{
      val parts = line.split(",")
      RowBayes(parts(0),parts(1))
    })

    //    textData.select("category","text").take(2).foreach(println(_))
    val splitDataRdd = srcRDD.randomSplit(Array(0.8,0.2),seed = 11L)
    var trainDF = splitDataRdd(0).toDF()
    var testDF = splitDataRdd(1).toDF()




//    trainData.select("text").take(2).foreach(println(_))

    //将特征词转换成数组
    var tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    var wordsData = tokenizer.transform(trainDF)
//    wordsData.select("words").take(2).foreach(println(_))

    //将转换成数组的特征进行TF处理成整型
    var hashTF = new HashingTF().setNumFeatures(25000).setInputCol("words").setOutputCol("Rawfeatures")
    var hashData = hashTF.transform(wordsData)
//    hashData.select("Rawfeatures").take(2).foreach(println(_))

    //计算TFIDF值
    var idr = new IDF().setInputCol("Rawfeatures").setOutputCol("features")
    var idfModel = idr.fit(hashData)
    var rescaledData = idfModel.transform(hashData)

//    rescaledData.select("features").take(2).foreach(println(_))

    var trainDataRdd = rescaledData.select("category","features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }.rdd

//    trainDataRdd.take(2).foreach(println(_))

    val model = NaiveBayes.train(trainDataRdd,lambda = 1.0,modelType = "multinomial")

    //测试数据集，做同样的特征表示及格式转换
    var testwordsData = tokenizer.transform(testDF)
    var testfeaturizedData = hashTF.transform(testwordsData)
    var testrescaledData = idfModel.transform(testfeaturizedData)

    var testDataRdd = testrescaledData.select("category","features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }.rdd

    //对测试数据集使用训练模型进行分类预测
    val testpredictionAndLabel = testDataRdd.map(p => (println(model.predict(p.features), p.label)))


  }
}

package Logistic_regression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.sql.SparkSession

/**
  * 垃圾邮件和正常邮件监测(英文)
  * Created by cluster on 2017/4/13.
  */
object GarbageEmail {
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

def main(args: Array[String]) {

  val conf = new SparkConf().setAppName("MLlib").setMaster("local[2]")
  val sc = new SparkContext(conf)

  val spam = sc.textFile("data/spam.txt")
  val normal = sc.textFile("data/normal.txt")

  // 创建一个HashingTF实例，将词转化为词频, 把邮件文本映射成包含25000特征的向量
  // HashingTF():特征哈希是一种处理高维数据的技术，经常应用在文本和分类数据集上
  val tf = new HashingTF(numFeatures = 25000)

  // 创建一个HashingTF实例，将词转化为词频, 把邮件文本映射成包含25000特征的向量
  // HashingTF():特征哈希是一种处理高维数据的技术，经常应用在文本和分类数据集上
  val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
  //打印 spamFeatures 显示如下：(密集向量表示)
  //(25000,[1263,3831,4287,5651,7607,7811,8008,8388,10373,11786,13537,14448,16302,18152,18333,18372,20468,20895,21600,22650],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0])
  val normalFeatures = normal.map(email => tf.transform(email.split(" ")))

  //创建LabeledPoint数据集 分别存放垃圾邮件(spam)和正常邮件(normal)
//  spamFeatures.collect().foreach( x => print(x + ","))
//  normalFeatures.collect().foreach { x => print(x + " ,") }

  // Create LabeledPoint datasets for positive (spam) and negative (ham) examples.
  val positiveExamples = normalFeatures.map(features => LabeledPoint(1,features))
  val negativeExamples = spamFeatures.map(features => LabeledPoint(0,features))
  // 数据集合并 返回一个新的数据集
  val trainingData = positiveExamples.union(negativeExamples)
  //因为逻辑回归是迭代算法  所以缓存训练数据的RDD
  trainingData.cache()

  //使用SGD算法运行逻辑回归
  val lrLearner = new LogisticRegressionWithSGD()
  val model = lrLearner.run(trainingData)

  //以垃圾邮件(0)和正常邮件(1)的例子分别进行测试进行测试
  val posTestExample = tf.transform("O M G GET cheap stuff by sending money to ...".split(" "))
  val negTestExample = tf.transform("Hi Dad, I started studying Spark the other ...".split(" "))

  val posTest1Example = tf.transform("I really wish well to all my friends.".split(" "))
  val posTest2Example = tf.transform("He stretched into his pocket for some money.".split(" "))
  val posTest3Example = tf.transform("He entrusted his money to me.".split(" "))
  val posTest4Example = tf.transform("Where do you keep your money?".split(" "))
  val posTest5Example = tf.transform("She borrowed some money of me.".split(" "))

  //首先使用，一样的HashingTF特征来得到特征向量，然后对该向量应用得到的模型
  println(s"Prediction for positive test example: ${model.predict(posTestExample)}")
  println(s"Prediction for negative test example: ${model.predict(negTestExample)}")

  println(s"posTest1Example for negative test example: ${model.predict(posTest1Example)}")
  println(s"posTest2Example for negative test example: ${model.predict(posTest2Example)}")
  println(s"posTest3Example for negative test example: ${model.predict(posTest3Example)}")
  println(s"posTest4Example for negative test example: ${model.predict(posTest4Example)}")
  println(s"posTest5Example for negative test example: ${model.predict(posTest5Example)}")

  sc.stop()


}
}

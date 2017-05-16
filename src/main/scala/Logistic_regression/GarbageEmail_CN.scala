package Logistic_regression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import util.JavaUtil

import scala.collection.mutable

/**
  * 垃圾邮件和正常邮件监测(英文)
  * Created by cluster on 2017/4/13.
  */
object GarbageEmail_CN {
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

  def pre(num: Double): Unit ={
    if (num == 1.0) println("这封邮件是-招聘邮件") else println("这封邮件是-正常邮件")
  }


  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("MLlib").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val job = sc.textFile("data/spam_CN.txt")
    val normal = sc.textFile("data/normal_CN.txt")
    val joblines = job.map(line => line.replaceAll("\uF0B2\t",""))
    val normalines = normal.flatMap(line => line.split(" "))
    //  joblines.foreach(println(_))
    import collection.JavaConverters._
    val joblineList = joblines.map(line => {val util = new JavaUtil
      util.getSplitWords(line)
    })
    val normalineList = normalines.map(line => {val util = new JavaUtil
      util.getSplitWords(line)
    })
    // 创建一个HashingTF实例，将词转化为词频, 把邮件文本映射成包含25000特征的向量
    // HashingTF():特征哈希是一种处理高维数据的技术，经常应用在文本和分类数据集上
    val tf = new HashingTF(numFeatures = 25000)
    // 创建一个HashingTF实例，将词转化为词频, 把邮件文本映射成包含25000特征的向量
    // HashingTF():特征哈希是一种处理高维数据的技术，经常应用在文本和分类数据集上
    val jobFeatures = joblineList.map(email => tf.transform(email.toArray()))
    //打印 spamFeatures 显示如下：(密集向量表示)
    //(25000,[1263,3831,4287,5651,7607,7811,8008,8388,10373,11786,13537,14448,16302,18152,18333,18372,20468,20895,21600,22650],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0])
    val normalFeatures = normalineList.map(email => tf.transform(email.toArray()))
    //创建LabeledPoint数据集 分别存放垃圾邮件(spam)和正常邮件(normal)
    val jobExamples = jobFeatures.map(features => LabeledPoint(1,features))
    val normalExamples = normalFeatures.map(features => LabeledPoint(0,features))
    // 数据集合并 返回一个新的数据集
    val trainingData = jobExamples.union(normalExamples)
    //  因为逻辑回归是迭代算法  所以缓存训练数据的RDD
    trainingData.cache()
    //  使用SGD算法运行逻辑回归
    val lrLearner = new LogisticRegressionWithSGD()
    val model = lrLearner.run(trainingData)
    //以垃圾邮件(0)和正常邮件(1)的例子分别进行测试进行测试
    val jobExample = tf.transform("负责,大数据,分析,系统,需求,分析,整体,架构,设计,负责,软硬件,实施,方案".split(","))
    val normalExample = tf.transform("加州,鹈鹕,湾监狱,囚犯,圣诞节,妈妈,苹果,十分,弟弟,孩子".split(","))
    //首先使用，一样的HashingTF特征来得到特征向量，然后对该向量应用得到的模型
    val num1 = model.predict(jobExample)
    val num2 = model.predict(normalExample)
    pre(num1)
    pre(num2)
    sc.stop()

  }
}

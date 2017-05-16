package Bayes

import java.io._
import java.util
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.sql.SQLContext

import org.apache.spark.{SparkContext, SparkConf}
import org.wltea.analyzer.core.{Lexeme, IKSegmenter}

/**
  * Created by cluster on 2017/4/26.
  */
object native_bayes_emtion {

  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  val wordLength = 0
  case class RowBayes(category: String,text: util.ArrayList[String])
  def main(args: Array[String]) {
    import org.apache.spark.sql
    val conf = new SparkConf().setAppName("emtion").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    //读取整个文件 获取数据
    val textRdd = sc.textFile("data/emtion.txt").map(line => line.split(",")).filter(line => line.length == 2)
//    textRdd.foreach(line => println(line(0),line(1)))
    //获取好评数据 评分 label为 5 的数据
    val postiveDataRdd = textRdd.filter(line => line(0)=="5").map(line => (0,line(1)))
//    println(postiveDataRdd.count())
    //获取差评数据 评分为：1，2，3
    val oneRdd = textRdd.filter(line => line(0)=="1")
    val twoRdd = textRdd.filter(line => line(0)=="2")
    val threeRdd = textRdd.filter(line => line(0)=="3")
    val negtiveRdd = oneRdd.union(twoRdd).union(threeRdd).map(line => (1,line(1)))
//    println(negtiveRdd.count())

    //获取和好评数量一样的数据,并且合并好评和差评
    val totalRdd = sc.parallelize(negtiveRdd.take(postiveDataRdd.count().asInstanceOf[Int])).union(postiveDataRdd).repartition(1)

//    totalRdd.foreach(line => println(line(0),line(1)))
    //对评论数据进行分词
    import sqlContext.implicits._
    val ikTotal = totalRdd.map(line => (line._1,ikSplitWord(line._2))).toDF()

    ikTotal.foreach(line => println(line))


    val hashTF = new HashingTF().setNumFeatures(25000)
    val totalHash = hashTF.transform(ikTotal)


    totalHash.foreach(println(_))



  }

  /**
    *
    * @param line
    * @return
    */
  def ikSplitWord(line: String): util.ArrayList[String] ={
    val words: util.ArrayList[String] = new util.ArrayList[String]()

    if (line == null || line.trim.length == 0) {
      return words
    }

    try {
      val is: InputStream = new ByteArrayInputStream(line.getBytes("UTF-8"))
      val seg: IKSegmenter = new IKSegmenter(new InputStreamReader(is), false)
      var lex: Lexeme = seg.next
      while (lex != null) {
        val word: String = lex.getLexemeText
        if (wordLength == 0 || word.length == wordLength) {
          words.add(word)
        }
        lex = seg.next
      }
    }
    catch {
      case e: UnsupportedEncodingException => {
        e.printStackTrace
      }
      case e: IOException => {
        e.printStackTrace
      }
    }

    words
  }
}

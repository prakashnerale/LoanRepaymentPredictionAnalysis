import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object ExploratoryAnalysis extends App {
  var conf = new SparkConf()
    .setAppName("prakash")
    .setMaster("local[5]")

  val sc = SparkContext.getOrCreate(conf)

  val spark = SparkSession.builder()
    .appName("Inclass ModelGenerator")
    .config("spark.sql.warehouse.dir", "D://BigData Spark//spark-warehouse")
    .getOrCreate()

  import spark.implicits._

  val raw = spark.read.text("data-files")
  
  raw.show()
  
  //val data = spark.read.option("inferSchema", true).csv("data-files").limit(1000)
  
  //data.show()

  case class Data(
    TARGET:              Int,
    NAME_CONTRACT_TYPE:  String,
    CODE_GENDER:         String,
    FLAG_OWN_CAR:        String,
    FLAG_OWN_REALTY:     String,
    CNT_CHILDREN:        Int,
    AMT_INCOME_TOTAL:    Double,
    AMT_CREDIT:          Double,
    AMT_ANNUITY:         Double,
    NAME_EDUCATION_TYPE: String)

  val df = raw.map(_.getString(0).split(","))
    .map(arr => Data(arr(0).toInt, arr(1).toString, arr(2).toString, arr(3).toString, arr(4).toString, arr(5).toInt, arr(6).toDouble, arr(7).toDouble, arr(8).toDouble, arr(9).toString))
    .toDF

  /**
   * Add a new column CREDIT_INCOME_PERCENT.
   * CREDIT_INCOME_PERCENT = AMT_CREDIT / AMT_INCOME_TOTAL
   *
   */
  val df1 = df.withColumn("CREDIT_INCOME_PERCENT", col("AMT_CREDIT") / col("AMT_INCOME_TOTAL"))
  //df1.show()

  /**
   * Load and Parse the Dataset using DataFrames.
   *
   */
  val columns = Seq("TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE",
    "FLAG_PHONE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION", "ORGANIZATION_TYPE", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12")

  val data = spark.read.option("inferSchema", true).csv("data-files").limit(1000).toDF(columns: _*).cache()

  //Exploratory Analysis

  // Calculate No. of loans falling into each Target with percentage

  //data.show()

  data.createOrReplaceTempView("data_view")

  data.groupBy("TARGET").count().withColumn("Percentage", col("count") * 100 / data.count()) //.show()

  spark.sql("""SELECT target, 
                      COUNT, 
                      COUNT * 100 / Sum(COUNT) OVER (ROWS BETWEEN unbounded preceding AND unbounded following) AS percentage 
                      FROM   (SELECT target, 
                                     COUNT(1) COUNT 
                              FROM   data_view 
                              GROUP  BY target) t 
            """).show()

  /**Number of missing values in each column*/
  //spark.sql("select sum(CAST((TARGET IS NULL) AS INT)) AS `TARGET` from data_view").show()

  val nullcounts = data.select(data.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*)
  val totcount = data.count
  nullcounts.first().toSeq.zip(data.columns).map(x => (x._1, "%1.2f".format(x._1.toString.toDouble * 100 / totcount), x._2)) //.foreach(println)

  /**
   * Describe days employed
   * Computes statistics for numeric columns, including count, mean, stddev, min, and max.
   *
   */
  data.select("DAYS_EMPLOYED").describe().show()

  spark.sql("select DAYS_EMPLOYED from data_view").describe() //.show()

  /**Describe days birth column*/
  val dfAge = data.withColumn("AGE", col("DAYS_BIRTH") / -365)
  dfAge.select("DAYS_BIRTH", "AGE").describe() //.show()

  /**
   * Dig deep into anomalies of DAY_EMPLOYED column. Calculate the number of wrong employment day column.
   */

  val anom = dfAge.filter(col("DAYS_EMPLOYED").equalTo(365243))
  val non_anom = dfAge.filter(col("DAYS_EMPLOYED").notEqual(365243))

  // Calculate anomalies and Non-anomalies default.
  val nonanomPer = 100 * non_anom.agg(avg(col("TARGET"))).first()(0).toString.toDouble
  val anomPer = 100 * anom.agg(avg(col("TARGET"))).first()(0).toString.toDouble
  println(f"The non-anomalies default on $nonanomPer%2.2f while anomalies default on $anomPer%2.2f ")

  /** Finally the number of wrong employment day column */
  val anomCount = anom.count
  print(f"There are $anomCount%d anomalous days of employment")

  // Create anomaly flag column
  val anomalyDf = dfAge.withColumn("DAYS_EMPLOYED_ANOM", col("DAYS_EMPLOYED").equalTo(365243))
  anomalyDf.show()

  //Replace anomaly value with 0
  val anomalyFlagDf = anomalyDf.withColumn("DAYS_EMPLOYED", when(col("DAYS_EMPLOYED") === 365243, 0).otherwise(col("DAYS_EMPLOYED"))) // if anom is 365243 convert to 0
  anomalyFlagDf.show()

  //Effect of age on repayment by binning the column and the generating pivot table
  anomalyFlagDf.select("AGE").describe().show()

  /**
   * Create new variables based on domain knowledge.
   *
   * 1. Calculate CREDIT_INCOME_PERCENT = AMT_CREDIT / AMT_INCOME_TOTAL
   *
   * 2. Calculate ANNUITY_INCOME_PERCENT = AMT_ANNUITY / AMT_INCOME_TOTAL
   *
   * 3. Calculate CREDIT_TERM = AMT_ANNUITY / AMT_CREDIT
   *
   * 4. DAYS_EMPLOYED_PERCENT = DAYS_EMPLOYED / DAYS_BIRTH
   *
   * 5. Add a new column "target"
   */

  val tmpDf1 = anomalyFlagDf.withColumn("CREDIT_INCOME_PERCENT", col("AMT_CREDIT") / col("AMT_INCOME_TOTAL"))

  val tmpDf2 = tmpDf1.withColumn("ANNUITY_INCOME_PERCENT", col("AMT_ANNUITY") / col("AMT_INCOME_TOTAL"))

  val tmpDf3 = tmpDf2.withColumn("CREDIT_TERM", col("AMT_ANNUITY") / col("AMT_CREDIT"))

  val tmpDf4 = tmpDf3.withColumn("DAYS_EMPLOYED_PERCENT", col("DAYS_EMPLOYED") / col("DAYS_BIRTH"))

  val newDf = tmpDf4.withColumn("label", col("TARGET"))

  /**
   * Convert string column with only 2 unique values to a column of label indices
   * to make the values readable for machine learning algorithm.
   */

  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.feature.StringIndexer

  val indexers = Array("NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY")
    .map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_Index"))
  val pipeline = new Pipeline().setStages(indexers)
  val df_r = pipeline.fit(newDf).transform(newDf)
  //df_r.show()

  /**
   * A one-hot encoder that maps a column of category indices to a column of binary vectors, with at most a single
   * one-value per row that indicates the input category index. For example with 5 categories, an input value of 2.0
   * would map to an output vector of [0.0, 0.0, 1.0, 0.0]. The last category is not included by default
   * (configurable via OneHotEncoder!.dropLast because it makes the vector entries sum up to one,
   * and hence linearly dependent. So an input value of 4.0 maps to [0.0, 0.0, 0.0, 0.0]
   *
   */
  import org.apache.spark.ml.feature.OneHotEncoder
  val indexers1 = Array("NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE")
    .map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_Index"))

  /**
   * Convert string column with values > 2 to one hot encoder
   * One hot encoding is a process by which categorical variables are converted into a form that 
   * could be provided to ML algorithms to do a better job in prediction.
   * 
   */
  val encoder = Array("NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE")
    .map(column => new OneHotEncoder().setInputCol(column + "_Index").setOutputCol(column + "_Vec"))

  val encoderPipeline = new Pipeline().setStages(indexers1 ++ encoder)
  val encoded = encoderPipeline.fit(df_r).transform(df_r)
  // encoded.show()

  //  Convert AGE column to bins (converting age in four categories)
  import org.apache.spark.ml.feature.Bucketizer
  val splits = Array(0, 25.0, 35.0, 55.0, 100.0)
  val bucketizer = new Bucketizer().setInputCol("AGE").setOutputCol("bucketedData").setSplits(splits)
  val bucketedData = bucketizer.transform(encoded)
  bucketedData.groupBy("bucketedData").pivot("TARGET").count().show() // bucketeddata is output column name

  
  //  Generate feature columns(discarded string only index columns)
  val feature_cols = Array(
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED",
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE",
    "FLAG_PHONE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "NAME_CONTRACT_TYPE_Index", "CODE_GENDER_Index",
    "FLAG_OWN_CAR_Index", "FLAG_OWN_REALTY_Index", "NAME_INCOME_TYPE_Vec", "NAME_EDUCATION_TYPE_Vec",
    "ORGANIZATION_TYPE_Vec", "AGE", "DAYS_EMPLOYED_ANOM", "bucketedData", "CREDIT_INCOME_PERCENT",
    "ANNUITY_INCOME_PERCENT", "CREDIT_TERM", "DAYS_EMPLOYED_PERCENT")

    
  //  Assemble features (assemble all features in one vector)
  import org.apache.spark.ml.feature.VectorAssembler
  val assembler = new VectorAssembler().setInputCols(feature_cols).setOutputCol("features")
  val output = assembler.transform(bucketedData)

  
  //  Train logistic Regression model (creating initializing and fitting model)
  import org.apache.spark.ml.classification.LogisticRegression
  val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
  val lrModel = lr.fit(output)
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  
  //  Get model Accuracy
  import org.apache.spark.sql.types._
  import org.apache.spark.mllib.evaluation.MulticlassMetrics
  
  val transformed = lrModel.transform(output)
  transformed.show()
  val results = transformed.select("prediction", "label").withColumn("label", col("label").cast(DoubleType))
  val predictionAndLabels = results.rdd.map(row => (row(0).toString.toDouble, row(1).toString.toDouble))
  val metrics = new MulticlassMetrics(predictionAndLabels)
  
  println("Confusion matrix:")
  println(metrics.confusionMatrix)
  
  val accuracy = metrics.accuracy 
  println ("Summary Statistics") 
  println (s"Accuracy = $accuracy")

}
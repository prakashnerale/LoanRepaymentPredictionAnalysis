import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{ OneHotEncoder, StringIndexer }
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object ModelGenerator {
  def main(args: Array[String]) {
    /* if (args.length < 1) {
      System.err.println("Usage: ModelGenerator <data path> <model persitence path>")
      System.exit(1)
    }*/

    var conf = new SparkConf()
      .setAppName("prakash")
      .setMaster("local[5]")

    var sc = SparkContext.getOrCreate(conf)

    var spark = SparkSession.builder()
    .appName("Inclass ModelGenerator")
    .config("spark.sql.warehouse.dir", "C://Users//prakash.chauhan//workspace//final-project//spark-warehouse")
    .getOrCreate()

    // data and model path     val dataPath = args(0) val modelPath = args(1)

    // load data
    val raw = spark.read.option("inferSchema", true).csv("data-files\\final_project")

    // Add header
    val columns = Seq(
      "TARGET",
      "NAME_CONTRACT_TYPE",
      "CODE_GENDER",
      "FLAG_OWN_CAR",
      "FLAG_OWN_REALTY",
      "CNT_CHILDREN",
      "AMT_INCOME_TOTAL",
      "AMT_CREDIT",
      "AMT_ANNUITY",
      "NAME_INCOME_TYPE",
      "NAME_EDUCATION_TYPE",
      "NAME_FAMILY_STATUS",
      "NAME_HOUSING_TYPE",
      "DAYS_BIRTH",
      "DAYS_EMPLOYED",
      "FLAG_MOBIL",
      "FLAG_EMP_PHONE",
      "FLAG_WORK_PHONE",
      "FLAG_CONT_MOBILE",
      "FLAG_PHONE",
      "CNT_FAM_MEMBERS",
      "REGION_RATING_CLIENT",
      "REGION_RATING_CLIENT_W_CITY",
      "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
      "ORGANIZATION_TYPE",
      "FLAG_DOCUMENT_2",
      "FLAG_DOCUMENT_3",
      "FLAG_DOCUMENT_4",
      "FLAG_DOCUMENT_5",
      "FLAG_DOCUMENT_6",
      "FLAG_DOCUMENT_7",
      "FLAG_DOCUMENT_8",
      "FLAG_DOCUMENT_9",
      "FLAG_DOCUMENT_10",
      "FLAG_DOCUMENT_11",
      "FLAG_DOCUMENT_12")

    val data = raw.limit(10000).toDF(columns: _*)
    data.cache()

    // Add age columns
    val dfAge = data.withColumn("AGE", col("DAYS_BIRTH") / (-365))

    // Add anomaly flag and replace it with 0
    val anomalyFlagDf = dfAge.withColumn("DAYS_EMPLOYED_ANOM", col("DAYS_EMPLOYED").equalTo(365243))
    val anomalyDf = anomalyFlagDf.withColumn("DAYS_EMPLOYED", when(col("DAYS_EMPLOYED") === 365243, 0).otherwise(col("DAYS_EMPLOYED")))

    // Rename column TARGET to label
    val labelDf = anomalyDf.withColumn("label", col("TARGET"))

    // create domain features
    val tmpDf1 = labelDf.withColumn("CREDIT_INCOME_PERCENT", col("AMT_CREDIT") / col("AMT_INCOME_TOTAL"))
    val tmpDf2 = tmpDf1.withColumn("ANNUITY_INCOME_PERCENT", col("AMT_ANNUITY") / col("AMT_INCOME_TOTAL"))
    val tmpDf3 = tmpDf2.withColumn("CREDIT_TERM", col("AMT_ANNUITY") / col("AMT_CREDIT"))
    val df = tmpDf3.withColumn("DAYS_EMPLOYED_PERCENT", col("DAYS_EMPLOYED") / col("DAYS_BIRTH"))

    // define columns that will be used as feature variables in model training
    val feature_cols = Array(
      "CNT_CHILDREN",
      "AMT_INCOME_TOTAL",
      "AMT_CREDIT",
      "AMT_ANNUITY",
      "DAYS_EMPLOYED",
      "FLAG_MOBIL",
      "FLAG_EMP_PHONE",
      "FLAG_WORK_PHONE",
      "FLAG_CONT_MOBILE",
      "FLAG_PHONE",
      "CNT_FAM_MEMBERS",
      "REGION_RATING_CLIENT",
      "REGION_RATING_CLIENT_W_CITY",
      "REG_REGION_NOT_LIVE_REGION",
      "REG_REGION_NOT_WORK_REGION",
      "FLAG_DOCUMENT_2",
      "FLAG_DOCUMENT_3",
      "FLAG_DOCUMENT_4",
      "FLAG_DOCUMENT_5",
      "FLAG_DOCUMENT_6",
      "FLAG_DOCUMENT_7",
      "FLAG_DOCUMENT_8",
      "FLAG_DOCUMENT_9",
      "FLAG_DOCUMENT_10",
      "FLAG_DOCUMENT_11",
      "FLAG_DOCUMENT_12",
      "NAME_CONTRACT_TYPE_Index",
      "CODE_GENDER_Index",
      "FLAG_OWN_CAR_Index",
      "FLAG_OWN_REALTY_Index",
      "NAME_INCOME_TYPE_Vec",
      "NAME_EDUCATION_TYPE_Vec",
      "ORGANIZATION_TYPE_Vec",
      "AGE",
      "DAYS_EMPLOYED_ANOM",
      "bucketedData",
      "CREDIT_INCOME_PERCENT",
      "ANNUITY_INCOME_PERCENT",
      "CREDIT_TERM",
      "DAYS_EMPLOYED_PERCENT")

    // Convert string to label index
    val indexers = Array("NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE")
      .map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_Index"))

    //val indexers = Array("CODE_GENDER","NAME_INCOME_TYPE").map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_In dex"))     println("==> Indexed")

    // convert string columns to binary columns
    val encoder = Array("NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE")
      .map(column => new OneHotEncoder().setInputCol(column + "_Index").setOutputCol(column + "_Vec"))

    // convert continuous variable to category
    val splits = Array(0, 25.0, 35.0, 55.0, 100.0)
    val bucketizer = new Bucketizer().setInputCol("AGE").setOutputCol("bucketedData").setSplits(splits)

    // Assemble features
    val assembler = new VectorAssembler().setInputCols(feature_cols).setOutputCol("features")

    // LogisticRegression Model
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    // Define pipeline
    //val pipeline = new Pipeline().setStages(Array(bucketizer) ++ indexers ++ encoder ++ Array(assembler, lr))
    val pipeline = new Pipeline().setStages(Array(bucketizer) ++ indexers ++ encoder ++ Array(assembler, lr))
    val model = pipeline.fit(df)

    model.transform(df).show()

    // save model     model.write.overwrite().save(modelPath)
    spark.stop()
  }
} 

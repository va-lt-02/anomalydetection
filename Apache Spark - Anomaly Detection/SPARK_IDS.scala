//import data

val dataWithoutHeader = spark.read.option("inferSchema", true).option("header", true).csv("/home/master/Desktop/GeneratedLabelledFlows/TrafficLabelling")


val data = dataWithoutHeader.toDF("Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std","Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std","Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min","Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min","Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min","Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length","Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance","FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length","Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes","Init_Win_bytes_forward", "Init_Win_bytes_backward","act_data_pkt_fwd", "min_seg_size_forward","Active Mean", "Active Std", "Active Max", "Active Min","Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label")

data.select("Label").groupBy("Label").count().orderBy($"count".desc).show(25)

val df = data.select("Source Port", "Destination Port", "Flow Duration", "Total Fwd Packets","Total Length of Fwd Packets", "FIN Flag Count", "SYN Flag Count", "PSH Flag Count", "ACK Flag Count","Label")

df.printSchema

//from string to int
//val df = data.selectExpr("cast(src_port as int) src_port", "cast(dst_port as int) dst_port", "cast(duration as int) duration", "cast(total_fwd_packets as int) total_fwd_packets","cast(total_length_fwd_packets as int) total_length_fwd_packets", "cast(FIN as int) FIN", "cast(SYN as int) SYN", "cast(PSH as int) PSH", "cast(ACK as int) ACK", "Label")
//df.printSchema

//Clustering

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().
setInputCols(df.columns.filter(_ != "Label")).
setOutputCol("featureVector")

val kmeans = new KMeans().
setPredictionCol("cluster").
setFeaturesCol("featureVector")

val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

assembler.setHandleInvalid("skip")

val pipelineModel = pipeline.fit(df)

val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

kmeansModel.clusterCenters.foreach(println)

val withCluster = pipelineModel.transform(df)

withCluster.select("cluster", "Label").
groupBy("cluster", "Label").count().
orderBy($"cluster", $"count".desc).
show(25)

//Choosing K
import org.apache.spark.sql.DataFrame
def clusteringScore0(data: DataFrame, k: Int): Double = {
val assembler = new VectorAssembler().
setInputCols(data.columns.filter(_ != "Label")).
setOutputCol("featureVector")
assembler.setHandleInvalid("skip")
val kmeans = new KMeans().
setSeed(Random.nextLong()).
setK(k).
setPredictionCol("cluster").
setFeaturesCol("featureVector")
val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
kmeansModel.computeCost(assembler.transform(data)) / data.count()
}
(20 to 100 by 20).map(k => (k, clusteringScore0(df, k))).
foreach(println)

//setMaxIter and setTol

def clusteringScore1(data: DataFrame, k: Int): Double = {
val assembler = new VectorAssembler().
setInputCols(data.columns.filter(_ != "Label")).
setOutputCol("featureVector")
val kmeans = new KMeans().
setSeed(Random.nextLong()).
setK(k).
setPredictionCol("cluster").
setFeaturesCol("featureVector").
setMaxIter(40).
setTol(1.0e-5)
val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
kmeansModel.computeCost(assembler.transform(data)) / data.count()
}
(20 to 100 by 20).map(k => (k, clusteringScore0(df, k))).
foreach(println)

//Feature Normalization
import org.apache.spark.ml.feature.StandardScaler

def clusteringScore2(data: DataFrame, k: Int): Double = {
val assembler = new VectorAssembler().
setInputCols(data.columns.filter(_ != "Label")).
setOutputCol("featureVector")
val scaler = new StandardScaler()
.setInputCol("featureVector")
.setOutputCol("scaledFeatureVector")
.setWithStd(true)
.setWithMean(false)
val kmeans = new KMeans().
setSeed(Random.nextLong()).
setK(k).
setPredictionCol("cluster").
setFeaturesCol("scaledFeatureVector").
setMaxIter(40).
setTol(1.0e-5)
val pipeline = new Pipeline().setStages(Array(assembler, scaler, kmeans))
val pipelineModel = pipeline.fit(data)
val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
}
(60 to 270 by 30).map(k => (k, clusteringScore2(df, k))).
foreach(println)
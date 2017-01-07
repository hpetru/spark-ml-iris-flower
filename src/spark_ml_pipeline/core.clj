(ns spark-ml-pipeline.core
  (:require [flambo.conf :as conf]
            [flambo.api :as f]
            [flambo.sql :as sql])

  (:import [org.apache.spark.sql.types StructType StructField Metadata DataTypes]
           [org.apache.spark.ml.feature VectorAssembler StringIndexer VectorIndexer]
           [org.apache.spark.ml.feature IndexToString]
           [org.apache.spark.ml Pipeline PipelineStage]
           [org.apache.spark.ml.evaluation MulticlassClassificationEvaluator]
           [org.apache.spark.ml.classification DecisionTreeClassifier LogisticRegression]
           [org.apache.spark.ml.tuning CrossValidator ParamGridBuilder]))

(defn build-sql-context
  []
  (let [c (-> (conf/spark-conf)
              (conf/master "local[*]")
              (conf/app-name "Iris Demo"))
        sc (f/spark-context c)]
    (sql/sql-context sc)))



(defonce sql-context (build-sql-context))

(def file-path
  "/Users/petru/Work/clojure/spark-ml-pipeline/resources/iris.csv")

(def dataset-schema
  (StructType.
    (into-array StructField
                [(StructField. "sepal_l" (DataTypes/DoubleType) false (Metadata/empty))
                 (StructField. "sepal_w" (DataTypes/DoubleType) false (Metadata/empty))
                 (StructField. "petal_l" (DataTypes/DoubleType) false (Metadata/empty))
                 (StructField. "petal_w" (DataTypes/DoubleType) false (Metadata/empty))
                 (StructField. "spicies" (DataTypes/StringType) false (Metadata/empty))])))


(def dataset
  (-> sql-context
      .read
      (.format "com.databricks.spark.csv")
      (.schema dataset-schema)
      (.load file-path)))

(def data-split
  (let [[train test] (.randomSplit dataset (double-array [0.7 0.3]))]
    [train test]))

(def train-dataset
  (get data-split 0))

(def test-dataset
  (get data-split 1))

(def features-assember
  (-> (VectorAssembler.)
      (.setInputCols (into-array String ["sepal_l" "sepal_w" "petal_l" "petal_w"]))
      (.setOutputCol "features")))

(def label-indexer
  (-> (StringIndexer.)
      (.setInputCol "spicies")
      (.setOutputCol "indexed_label")
      (.fit dataset)))

(def features-indexer
  (-> (VectorIndexer.)
      (.setInputCol "features")
      (.setOutputCol "indexed_features")
      (.setMaxCategories 3)))

(def classifier
  (-> (DecisionTreeClassifier.)
      (.setLabelCol "indexed_label")
      (.setFeaturesCol "indexed_features")
      (.setPredictionCol "prediction")))

(def label-converter
  (-> (IndexToString.)
      (.setInputCol "prediction")
      (.setOutputCol "predicted_label")
      (.setLabels (.labels label-indexer))))

(def pipeline-stages
  (into-array PipelineStage
              [features-assember
               label-indexer
               features-indexer
               classifier
               label-converter]))

(def pipeline
  (-> (Pipeline.)
      (.setStages pipeline-stages)))

(def parameter-grid
  (-> (ParamGridBuilder.)
      (.addGrid (.maxDepth classifier) (int-array [5 10 30]))
      (.addGrid (.maxBins classifier) (int-array [10 100 1000]))
      (.build)))

(def evaluator
  (-> (MulticlassClassificationEvaluator.)
      (.setLabelCol "indexed_label")
      (.setPredictionCol "prediction")))

(def cross-validator
  (-> (CrossValidator.)
      (.setEstimator pipeline)
      (.setEvaluator evaluator)
      (.setEstimatorParamMaps parameter-grid)
      (.setNumFolds 2)))

(def cross-validator-model
  (.fit cross-validator train-dataset))

(def model
  (.bestModel cross-validator-model))

; ---------------

(def predictions
  (.transform model test-dataset))

(def accuracy
  (.evaluate evaluator predictions))

(defn -main
  [& args]
  (do
    (println (str "Test error: " (- 1.0 accuracy)))
    (println (str "Accuracy: " accuracy))))

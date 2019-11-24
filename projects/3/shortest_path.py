from pyspark import SparkConf
from pyspark.sql import SparkSession
import os
import sys
import pyspark.sql.functions as f
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')


def shortest_path(v_from, v_to, twitter, max_path_length=10):
    temp = twitter.filter(twitter.follower_id == v_from)
    graph = temp.select(temp.follower_id.alias("vertex_from"), temp.follower_id.alias("path"),\
                    temp.user_id.alias("vertex_to"))
    for i in range(max_path_length):
        graph = graph.join(twitter, graph.vertex_to == twitter.follower_id).select(graph.vertex_from, 
                                                                       f.concat_ws(',', graph.path, graph.vertex_to).alias("path"),
                                                                       twitter.user_id.alias("vertex_to"))
        if graph.filter(graph.vertex_to == v_to).count() > 0:
            res = graph.filter(graph.vertex_to == v_to).select(f.concat_ws(',', graph.path, graph.vertex_to).alias("path"))
            break
    return res
    
schema = StructType(fields=[StructField("user_id", IntegerType()), StructField("follower_id", IntegerType())])
v_from = sys.argv[1]
v_to = sys.argv[2]
path_to_dataset = sys.argv[3]
output_directory = sys.argv[4]
#v_from = 12
#v_to = 34
#path_to_dataset = "/datasets/twitter/twitter_sample.tsv"
#output_directory = "hw3_output"
twitter = spark.read\
          .schema(schema)\
          .format("csv")\
          .option("sep", "\t")\
          .load(path_to_dataset)

result = shortest_path(v_from, v_to, twitter, max_path_length=10)
result.select("path").write.mode("overwrite").text(output_directory)
#result.show(truncate=False)

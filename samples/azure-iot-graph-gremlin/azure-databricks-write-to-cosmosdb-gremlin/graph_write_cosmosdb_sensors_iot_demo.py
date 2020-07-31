# Databricks notebook source
# MAGIC %md # Writing GraphFrames to Azure Cosmos DB Gremlin API
# MAGIC 
# MAGIC Code adapted by Caio Moreno <BR>
# MAGIC 
# MAGIC Original code: https://github.com/syedhassaanahmed/databricks-notebooks/blob/master/graph_write_cosmosdb.py <BR>
# MAGIC 
# MAGIC This notebook is based on the `GraphFrames` example [specified here](https://graphframes.github.io/user-guide.html#tab_python_0). It requires [graphframes](https://spark-packages.org/package/graphframes/graphframes) and [azure-cosmosdb-spark (uber jar)](https://github.com/Azure/azure-cosmosdb-spark#using-databricks-notebooks) libraries to be uploaded and attached to the cluster. **Scala version** of this notebook can be [found here](https://github.com/syedhassaanahmed/databricks-notebooks/blob/master/graphWriteCosmosDB.scala)

# COMMAND ----------

from pyspark.sql.functions import lit

v = sqlContext.createDataFrame([
  ("1", "Construction Site 1", 1, "London"),
  ("2", "Construction Site 2", 3, "Madrid"),
  ("3", "Construction Site 3", 3, "Dublin"),
  ("4", "Construction Site 4", 0, "Sao Paulo"),
  ("5", "Construction Site 5", 10, "Rome"),
  ("6", "Construction Site 6", 12, "Rio de Janeiro"),
  ("7", "Construction Site 7", 14, "Hong Kong")
], ["id", "sensorname", "sensorage", "sensorcitylocation"]) \
.withColumn("entity", lit("sensor"))

# COMMAND ----------

e = sqlContext.createDataFrame([
  ("1", "2", "connected"),
  ("1", "3", "connected"),
  ("1", "4", "connected"),
  ("1", "5", "connected"),
  ("1", "6", "connected"),
  ("1", "7", "connected"),
  ("2", "4", "connected"),
  ("2", "6", "connected"),
  ("2", "7", "connected")  
], ["src", "dst", "relationship"])

# COMMAND ----------

from graphframes import GraphFrame
g = GraphFrame(v, e)

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

display(g.edges)

# COMMAND ----------

# MAGIC %md ## Convert Vertices and Edges to Cosmos DB internal format
# MAGIC Cosmos DB Gremlin API internally keeps a JSON document representation of Edges and Vertices [as explained here](https://github.com/LuisBosquez/azure-cosmos-db-graph-working-guides/blob/master/graph-backend-json.md). Also `id` in Cosmos DB is [part of the resource URI](https://github.com/Azure/azure-cosmosdb-dotnet/issues/35#issuecomment-121009258) and hence must be URL encoded.

# COMMAND ----------

from pyspark.sql.types import StringType
from urllib.parse import quote

def urlencode(value):
  return quote(value, safe="")

udf_urlencode = udf(urlencode, StringType())

# COMMAND ----------

def to_cosmosdb_vertices(dfVertices, labelColumn, partitionKey = ""):
  dfVertices = dfVertices.withColumn("id", udf_urlencode("id"))
  
  columns = ["id", labelColumn]
  
  if partitionKey:
    columns.append(partitionKey)
  
  columns.extend(['nvl2({x}, array(named_struct("id", uuid(), "_value", {x})), NULL) AS {x}'.format(x=x) \
                for x in dfVertices.columns if x not in columns])
 
  return dfVertices.selectExpr(*columns).withColumnRenamed(labelColumn, "label")

# COMMAND ----------

cosmosDbVertices = to_cosmosdb_vertices(g.vertices, "entity")
display(cosmosDbVertices)

# COMMAND ----------

from pyspark.sql.functions import concat_ws, col

def to_cosmosdb_edges(g, labelColumn, partitionKey = ""): 
  dfEdges = g.edges
  
  if partitionKey:
    dfEdges = dfEdges.alias("e") \
      .join(g.vertices.alias("sv"), col("e.src") == col("sv.id")) \
      .join(g.vertices.alias("dv"), col("e.dst") == col("dv.id")) \
      .selectExpr("e.*", "sv." + partitionKey, "dv." + partitionKey + " AS _sinkPartition")

  dfEdges = dfEdges \
    .withColumn("id", udf_urlencode(concat_ws("_", col("src"), col(labelColumn), col("dst")))) \
    .withColumn("_isEdge", lit(True)) \
    .withColumn("_vertexId", udf_urlencode("src")) \
    .withColumn("_sink", udf_urlencode("dst")) \
    .withColumnRenamed(labelColumn, "label") \
    .drop("src", "dst")
  
  return dfEdges

# COMMAND ----------

cosmosDbEdges = to_cosmosdb_edges(g, "relationship")
display(cosmosDbEdges)

# COMMAND ----------

# MAGIC %md ## Make sure to use the [Cosmos DB https endpoint](https://docs.microsoft.com/en-us/azure/cosmos-db/how-to-use-regional-gremlin#portal-endpoint-discovery) and **NOT** the `wss://` endpoint

# COMMAND ----------

cosmosDbConfig = {
  "Endpoint" : "https://<COSMOSDB_ENDPOINT>.documents.azure.com:443/",
  "Masterkey" : "<COSMOSDB_PRIMARYKEY>",
  "Database" : "<DATABASE>",
  "Collection" : "<COLLECTION>",
  "Upsert" : "true"
}

cosmosDbFormat = "com.microsoft.azure.cosmosdb.spark"

cosmosDbVertices.write.format(cosmosDbFormat).mode("append").options(**cosmosDbConfig).save()
cosmosDbEdges.write.format(cosmosDbFormat).mode("append").options(**cosmosDbConfig).save()

# COMMAND ----------



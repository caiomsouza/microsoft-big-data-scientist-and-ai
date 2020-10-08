# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC This sample code (Python) will demo how to consume Kafka topics using Azure Databricks (Spark), Confluent Cloud (Kafka) running on Azure, Schema Registry and AVRO format.
# MAGIC 
# MAGIC ## Set up the environment.

# COMMAND ----------

# MAGIC %md
# MAGIC * You must have a Confluent cluster, an API Key and secret, a Schema Registry, an API Key and secret for the registry, and a topic
# MAGIC * Do a pip install of this: confluent-kafka[avro,json,protobuf]>=1.4.2
# MAGIC * Notebooks must be detached and re-attached before they can see new libraries
# MAGIC * For production use you'll need the pip install in an init script

# COMMAND ----------

confluentBootstrapServers = "CHANGE_HERE"
#confluentApiKey = dbutils.secrets.get(scope = "confluentTest", key = "api-key")
#confluentSecret = dbutils.secrets.get(scope = "confluentTest", key = "secret")
#confluentRegistryApiKey = dbutils.secrets.get(scope = "confluentTest", key = "registry-api-key")
#confluentRegistrySecret = dbutils.secrets.get(scope = "confluentTest", key = "registry-secret")
confluentApiKey = "CHANGE_HERE"
confluentSecret = "CHANGE_HERE"
confluentRegistryApiKey = "CHANGE_HERE"
confluentRegistrySecret = "CHANGE_HERE"
confluentTopicName = "CHANGE_HERE"
schemaRegistryUrl = "CHANGE_HERE"


# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up the client for the Schema Registry

# COMMAND ----------

from confluent_kafka.schema_registry import SchemaRegistryClient

schema_registry_conf = {
    'url': schemaRegistryUrl,
    'basic.auth.user.info': '{}:{}'.format(confluentRegistryApiKey, confluentRegistrySecret)}

schema_registry_client = SchemaRegistryClient(schema_registry_conf)

# COMMAND ----------

import pyspark.sql.functions as fn
from pyspark.sql.avro.functions import from_avro

keyRestResponseSchema = schema_registry_client.get_latest_version(confluentTopicName + "-key").schema
confluentKeySchema = keyRestResponseSchema.schema_str
valueRestResponseSchema = schema_registry_client.get_latest_version(confluentTopicName + "-value").schema
confluentValueSchema = valueRestResponseSchema.schema_str

# Set the option for how to fail - either stop on the first failure it finds (FAILFAST) or just set corrupt data to null (PERMISSIVE)
#fromAvroOptions = {"mode":"FAILFAST"}
fromAvroOptions= {"mode":"PERMISSIVE"}

AvroDF = ( 
  spark
  .readStream
  .format("kafka")
  .option("kafka.bootstrap.servers", confluentBootstrapServers)
  .option("kafka.security.protocol", "SASL_SSL")
  .option("kafka.sasl.jaas.config", "kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required username='{}' password='{}';".format(confluentApiKey, confluentSecret))
  .option("kafka.ssl.endpoint.identification.algorithm", "https")
  .option("kafka.sasl.mechanism", "PLAIN")
  .option("subscribe", confluentTopicName)
  .option("startingOffsets", "earliest")  
  .load()
  .withColumn('fixedKey', fn.expr("substring(key, 6, length(key)-5)"))
  .withColumn('fixedValue', fn.expr("substring(value, 6, length(value)-5)"))
  .select(from_avro('fixedKey',confluentKeySchema,fromAvroOptions).alias('parsedKey'), from_avro('fixedValue', confluentValueSchema,fromAvroOptions).alias('parsedValue'))
)

# COMMAND ----------

display(AvroDF)

# COMMAND ----------

# Create a DataFrame that blows out the parsedValue into the three Skechers columns
AvroDFCurated = AvroDF.select("parsedValue.SOSAresultTime", "parsedValue.SOSASensors.UAID", "parsedValue.SOSASensors.SOSAhasResult.numericValue")

# COMMAND ----------

display(AvroDFCurated)

# COMMAND ----------

AvroDFCurated.createOrReplaceTempView("loadtable")

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from loadtable

# COMMAND ----------

# MAGIC %sql
# MAGIC --clean table load
# MAGIC CREATE OR REPLACE TEMPORARY VIEW cleanloadtable as SELECT REPLACE( REPLACE(CAST(numericValue AS VARCHAR(50)), '[', '' ), ']', '' ) AS loadValue, REPLACE( REPLACE( REPLACE(CAST(UAID AS VARCHAR(50)), '[', '' ), ']', '' ), '"', '' ) AS deviceID, SOSAresultTime AS loadTimestamp FROM loadtable

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from cleanloadtable

# COMMAND ----------



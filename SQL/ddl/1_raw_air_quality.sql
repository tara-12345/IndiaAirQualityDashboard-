--1_raw_air_quality.sql

-- Creates the raw table for the AQ measurements.
-- Columns match the CSV schema from the OpenAQ S3 archive exactly.

CREATE TABLE IF NOT EXISTS raw.air_quality (
	--identifiers
	locationid BIGINT,
	location_id BIGINT,
	sensors_id BIGINT,

	--location information 
	"location" VARCHAR,
	lat DOUBLE,
	lon DOUBLE,

	--measurement
	"datetime" TIMESTAMP,
	"parameter" VARCHAR,

--partitioning (from S3 hive path)
	units VARCHAR,
	"value" DOUBLE,
	"month" VARCHAR,
	"year" BIGINT,

	--pipeline metadata
	ingestion_datetime TIMESTAMP
);
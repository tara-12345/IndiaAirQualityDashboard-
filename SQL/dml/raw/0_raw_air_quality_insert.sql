
--0_raw_air_quality_insert.sql

-- Inserts raw air quality readings from a single CSV file into raw.air_quality.
SELECT 
    --identifiers
    location_id,
    locationid,
    sensors_id,

    --location info
    "location",
    lat,
    lon,

    --measurement
    "datetime",
    "parameter",
    units,
    "value",

    --partitioning (from S3 hive path)
    "month",
    "year",

    --pipeline metadata (from the pipeline, not present in the source CSV)
    current_timestamp AS ingestion_datetime

FROM read_csv_auto(
   -- Reads directly from S3 without downloading the file first.
    '{{ data_file_path }}', --filled in at extraction
    filename=true,
    hive_partitioning=1, --reads year/month/locationid from the S3 folder structure
    union_by_name=true -- matches columns by name not position, added to debu when CSV schemas vary slightly.
);


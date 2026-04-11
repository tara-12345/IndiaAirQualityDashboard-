-- 01_presentation_fast.sql
-- Builds fast presenation layer, prevents crashing with large database. 
 
--drop existing tables.
DROP TABLE IF EXISTS presentation.daily_air_quality_stats;
DROP TABLE IF EXISTS presentation.daily_city_wide_core;
DROP TABLE IF EXISTS presentation.latest_param_values_per_location;
DROP TABLE IF EXISTS presentation.dim_locations;
 
 
-- One row per monitoring station.
CREATE OR REPLACE TABLE presentation.dim_locations AS
SELECT
  location_id,
  ANY_VALUE(location) AS location,
  AVG(lat) AS lat,
  AVG(lon) AS lon,
  MIN(CAST(datetime AS DATE)) AS first_date,
  MAX(CAST(datetime AS DATE)) AS last_date,
  COUNT(DISTINCT CAST(datetime AS DATE)) AS n_days
FROM raw.air_quality
WHERE lat IS NOT NULL AND lon IS NOT NULL
GROUP BY location_id;
 
 
-- Daily aggregates per location+parameter. Main table queried by the dashboard.
-- Joins dim_locations so lat/lon/location name are always consistent.
CREATE OR REPLACE TABLE presentation.daily_air_quality_stats AS
SELECT
  r.location_id,
  l.location,
  CAST(r.datetime AS DATE) AS measurement_date,
  l.lat,
  l.lon,
  r.parameter,
  ANY_VALUE(r.units) AS units,
  AVG(r.value) AS average_value,
  MEDIAN(r.value) AS median_value,
  MAX(r.value) AS max_value,
  MIN(r.value) AS min_value,
  COUNT(*) AS n_obs
FROM raw.air_quality r
JOIN presentation.dim_locations l ON r.location_id = l.location_id
WHERE r.value IS NOT NULL
  AND r.value >= 0
  AND r.parameter IS NOT NULL
GROUP BY
  r.location_id,
  l.location,
  CAST(r.datetime AS DATE),
  l.lat,
  l.lon,
  r.parameter;
 
 
-- Most recent reading per location+parameter. Used for the live map view.
-- ROW_NUMBER orders by datetime so rn=1 is always the latest record.
CREATE OR REPLACE TABLE presentation.latest_param_values_per_location AS
SELECT location_id, location, lat, lon, parameter, value, units, datetime
FROM (
  SELECT
    r.location_id,
    l.location,
    l.lat,
    l.lon,
    r.parameter,
    r.value,
    r.units,
    r.datetime,
    ROW_NUMBER() OVER (
      PARTITION BY r.location_id, r.parameter
      ORDER BY r.datetime DESC
    ) AS rn
  FROM raw.air_quality r
  JOIN presentation.dim_locations l ON r.location_id = l.location_id
  WHERE r.value IS NOT NULL
    AND r.parameter IS NOT NULL
) t
WHERE rn = 1;
 
 
-- Daily wide table with one column per pollutant.
-- Avoids repeats when computing correlations between parameters.
CREATE OR REPLACE TABLE presentation.daily_city_wide_core AS
SELECT
  location,
  measurement_date,
  MAX(CASE WHEN parameter = 'pm25' THEN average_value END) AS pm25,
  MAX(CASE WHEN parameter = 'pm10' THEN average_value END) AS pm10,
  MAX(CASE WHEN parameter = 'no2' THEN average_value END) AS no2,
  MAX(CASE WHEN parameter = 'o3' THEN average_value END) AS o3,
  MAX(CASE WHEN parameter = 'co' THEN average_value END) AS co,
  MAX(CASE WHEN parameter = 'so2' THEN average_value END) AS so2,
  MAX(CASE WHEN parameter = 'temperature' THEN average_value END) AS temperature,
  MAX(CASE WHEN parameter = 'relativehumidity' THEN average_value END) AS relativehumidity
FROM presentation.daily_air_quality_stats
GROUP BY location, measurement_date;
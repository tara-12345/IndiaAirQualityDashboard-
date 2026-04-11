--0_schemas.sql

-- Creates the two schemas used across the pipeline.
-- 'raw' will hold data exactly as it is
-- 'presentation' will hold cleaned tables that the dashboard queries directly.


CREATE SCHEMA IF NOT EXISTS 'raw';
CREATE SCHEMA IF NOT EXISTS 'presentation';
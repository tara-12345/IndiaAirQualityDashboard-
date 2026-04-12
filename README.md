# IndiaAirQualityDashboard-
Parameters covered
PM₂.₅, PM₁₀, NO₂, O₃, SO₂, CO, temperature, relative humidity, wind speed, wind direction

Project structure
├── app2.py                     # Streamlit dashboard
├── SQL/
│   ├── ddl/                    # Schema definitions (raw + presentation)
│   └── dml/
│       ├── raw/                # Extraction insert template
│       └── presentation/       # Transformation SQL files (run in sorted order)
├── extraction.py               # OpenAQ S3 extraction pipeline
├── transformation.py           # Runs SQL transformation files
├── database_manager.py         # Schema initialisation and DB utilities
├── get_locations.py            # Queries OpenAQ v3 API to retrieve Indian station IDs
├── locations.json              # 723 Indian station IDs from OpenAQ v3 API
└── requirements.txt

Data pipeline
OpenAQ S3 archive
│

extraction.py       ← iterates 723 Indian stations × monthly files
│                 reads compressed CSVs directly into DuckDB via read_csv_auto
│                 runs in parallel across 8 worker threads

raw.air_quality     ← one row per measurement, schema exactly as received
│

transformation.py   ← runs SQL files in sorted order
│
├─ presentation.dim_locations                    (one row per station, coordinates, date range)
├─ presentation.daily_air_quality_stats          (daily mean/max/min per station + pollutant)
└─ presentation.latest_param_values_per_location (most recent reading per station + pollutant)
│

app2.py             ← Streamlit dashboard, read-only DuckDB connection

Bash Scripts start - finish: 
Requirements 
pip install -r requirements.txt


Running the pipeline

 1. Initialise the database schema
python database_manager.py
python database_manager.py --create \
  --database-path air_quality_full.db \
  --ddl-query-parent-dir sql/

# Destroy the database (useful after extraction mistakes)
python database_manager.py --destroy \
  --database-path air_quality_full.db

  2. Get station IDs
python get_locations.py \
  --secrets_path secrets.json \
  --output_path locations.json \
  --country IN

4. Extract data from OpenAQ (Start and end date in YYYY-MM)
caffeinate -i python extraction.py \
  --locations_file_path locations.json \
  --start_date <STARTDATE> \
  --end_date <ENDDATE> \
  --database_path air_quality_full.db \
  --extract_query_template_path SQL/dml/raw/0_raw_air_quality_insert.sql \
  --source_base_path s3://openaq-data-archive/records/csv.gz

5. Run transformations to build the presentation layer
python python transform.py --database_path app/air_quality_full.db --query_directory sql/transforms
Missing files are logged and skipped automatically.

Launching the dashboard
streamlit run app/app.py

""" caffeinate -i python extraction.py \
  --locations_file_path locations.json \
  --start_date 2024-01 \
  --end_date 2025-12 \
  --database_path air_quality_full.db \
  --extract_query_template_path SQL/dml/raw/0_raw_air_quality_insert.sql \
  --source_base_path s3://openaq-data-archive/records/csv.gz"""

import argparse
import json
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from jinja2 import Template
from duckdb import IOException
from database_manager import open_db, close_db, read_file

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--locations_file_path", required=True)
parser.add_argument("--start_date", required=True, help="YYYY-MM")
parser.add_argument("--end_date", required=True, help="YYYY-MM")
parser.add_argument("--extract_query_template_path", required=True)
parser.add_argument("--database_path", required=True)
parser.add_argument("--source_base_path", required=True)
args = parser.parse_args()

# load location ids from json
with open(args.locations_file_path) as f:
    location_ids = [str(k) for k in json.load(f).keys()]

# load sql insert template once
sql_template = read_file(args.extract_query_template_path)

start = datetime.strptime(args.start_date, "%Y-%m")
end = datetime.strptime(args.end_date, "%Y-%m")

conn = open_db(args.database_path)

# loop every location + month combination
for loc_id in location_ids:
    date = start
    while date <= end:
        path = f"{args.source_base_path}/locationid={loc_id}/year={date.year}/month={str(date.month).zfill(2)}/*"
        query = Template(sql_template).render(data_file_path=path)
        try:
            conn.execute(query)
            logging.info(f"Extracted: {path}")
        except IOException:
            # not every location/month will have data
            logging.warning(f"Not found, skipping: {path}")
        date += relativedelta(months=1)

close_db(conn)
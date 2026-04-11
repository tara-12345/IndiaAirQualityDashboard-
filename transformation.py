import argparse
import logging
from database_manager import open_db, close_db, read_file, get_sql_files

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--database_path", required=True)
parser.add_argument("--query_directory", required=True)
args = parser.parse_args()

conn = open_db(args.database_path)

for path in get_sql_files(args.query_directory):
    conn.execute(read_file(path))
    logging.info(f"Executed: {path}")

close_db(conn)
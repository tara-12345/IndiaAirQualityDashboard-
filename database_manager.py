import os
import argparse

import duckdb


def open_db(location):
    # Connect to the DuckDB database file at the given path
    conn = duckdb.connect(location)

    # httpfs lets DuckDB read files directly from S3, no download. 
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")

    # public acces S3 archive. 
    conn.execute("SET s3_region='us-east-1';")
    conn.execute("SET s3_access_key_id='';")
    conn.execute("SET s3_secret_access_key='';")

    return conn


def close_db(conn):
    # Always close the connection to prevent double
    conn.close()


def get_sql_files(directory):
    # Find sql files. 
    matches = []

    for dirpath, _, filenames in os.walk(directory):
        for name in filenames:
            if name.endswith(".sql"):
                matches.append(os.path.join(dirpath, name))
    return sorted(matches)


def read_file(filepath):
    with open(filepath, "r") as f:
        return f.read()


def create_db(db_location, scripts_dir):
    files = get_sql_files(scripts_dir)
    conn = open_db(db_location)

    # Run each DDL script in order to build up the schema
    for fp in files:
        conn.execute(read_file(fp))

    close_db(conn)


def drop_db(db_location):
    if os.path.exists(db_location):
        os.remove(db_location)


def main():
    parser = argparse.ArgumentParser()

    # user must pass exactly one of --create or --destroy
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--destroy", action="store_true") #useful in extraction mistakes. 

    parser.add_argument("--database-path", required=True)
    parser.add_argument("--ddl-query-parent-dir")  # Only needed for --create

    args = parser.parse_args()

    if args.create:
        # Validate manually since --ddl-query-parent-dir is optional at parser level
        if not args.ddl_query_parent_dir:
            print("--ddl-query-parent-dir is required with --create")
            return
        create_db(db_location=args.database_path, scripts_dir=args.ddl_query_parent_dir)

    elif args.destroy:
        drop_db(db_location=args.database_path)


if __name__ == "__main__":
    main()
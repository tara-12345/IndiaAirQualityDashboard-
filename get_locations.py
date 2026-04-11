import json
import argparse
import logging
from openaq import OpenAQ

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--secrets_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--country", required=True, help="ISO country code e.g. IN")
args = parser.parse_args()

with open(args.secrets_path) as f:
    secrets = json.load(f)

client = OpenAQ(api_key=secrets["openaq-api-key"])

locations_info = {}
page = 1

while True:
    results = client.locations.list(iso=args.country, limit=1000, page=page)
    for location in results.results:
        locations_info[str(location.id)] = location.name
    logging.info(f"Page {page}: fetched {len(results.results)} locations (total so far: {len(locations_info)})")
    if len(results.results) < 1000:
        break
    page += 1

with open(args.output_path, "w", encoding="utf-8") as f:
    json.dump(locations_info, f, ensure_ascii=False, indent=4)

logging.info(f"Done. Saved {len(locations_info)} locations to {args.output_path}")

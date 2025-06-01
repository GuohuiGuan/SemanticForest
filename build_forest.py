import argparse
import time
from pipeline_forest import CreateForest

parser = argparse.ArgumentParser(description="Graph processing script")
parser.add_argument("--data_type", type=str, required=True, help="Dataset name (e.g., kaggle, harvard)")
parser.add_argument("--run_id", type=int, default=None, help="Run ID")
args = parser.parse_args()

data_type = args.data_type
run_id = args.run_id

print(f"Building Semantic Forest for {data_type}")
start_time = time.time()
creator = CreateForest(data_type=data_type, run_id=run_id)
creator.build_forest()
end_time = time.time()

print(f"Semantic Forest Time taken: {(end_time - start_time) / 60:.2f} minutes")

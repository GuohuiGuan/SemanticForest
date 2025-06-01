# Portions of this code are adapted from a project originally developed by Two Sigma Open Source, LLC,
# which is licensed under the Apache License, Version 2.0.
# 
# The original license and copyright notice are preserved below as required:
#
# Copyright 2024 Two Sigma Open Source, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import time
import json
import pickle
import hashlib
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from forest_utils import get_and_cluster_gfsts, build_semantic_forest, print_semantic_tree_json, print_semantic_tree, save_semantic_forest
from graph_construction import load_clean_graph, get_nodes_by_node_type, NodeType
from ray_cmds import code_compiling_func, Params, KeyValueStore, text_generation_func
from prompts.prompt_cluster_to_tree_structure import cluster_to_tree_structure_prompt

import ray

class CreateForest():
    def __init__(self, data_type, run_id=None):
        self.data_type = data_type
        self.run_id = run_id
        self.table_dir = f'assets/{data_type}/tables'
        self.data_store_dir = f'assets/{data_type}/forest/{data_type}_semantic_forest{f"_{run_id}" if run_id else ""}'
        self.key_value_store_name = f'{self.data_store_dir}/kv_store.pkl'
        self.tree_data_name = f'{self.data_store_dir}/{data_type}_tree_extraction.json'
        self.cluster_groups_name = f'{self.data_store_dir}/{self.data_type}_cluster_groups.pkl'
        self.embedding_cache_name = f'{self.data_store_dir}/embedding_cache.pkl'
        if not os.path.exists(self.data_store_dir):
            os.makedirs(self.data_store_dir)

        self.llm_params = Params(MAX_TOKENS=5000, USE_CACHE=False, USE_LARGE=False)

        self.graph_name = f'{self.data_store_dir}/{data_type}_graph.pkl'
        self.semantic_forest_name = f'{self.data_store_dir}/{data_type}_semantic_forest.pkl'
        self.total_tokens = defaultdict(int)
        self.api_calls = defaultdict(int)
        self.stats_path = f'assets/forest_stats.csv'
        os.makedirs('assets', exist_ok=True)

    def get_kv_store_actor(self):
        """
        To Cache GPT calls, we use the KV Store class, which is bound to a Ray Actor for parallelization.

        :return: kv store actor
        """
        if not os.path.exists(self.key_value_store_name):
            pickle.dump({}, open(self.key_value_store_name, 'wb'))

        return KeyValueStore.remote(self.key_value_store_name, clear_cache=False)
    
    def add_token_count(self, prompt, call_type: str) -> None:
        from prompt_utils import get_model, count_tokens_in_request
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)
        token_count = count_tokens_in_request(get_model(), prompt)
        self.total_tokens[call_type] += token_count
        self.api_calls[call_type] += 1
        
    def log_stats(self, elapsed_time, num_trees, num_total_nodes):
        """
        Log semantic forest statistics and safely update or insert based on (data_type, run_id).
        """
        new_row = {
            "data_type": self.data_type,
            "run_id": self.run_id or 0,
            "time": elapsed_time,
            "number_of_trees": num_trees,
            "number_of_nodes": num_total_nodes,
            "api_calls_total": sum(self.api_calls.values()),
            "tree_generation_calls": self.api_calls["tree_structure_call"],
            "tree_generation_tokens": self.total_tokens["tree_structure_call"],
        }

        key_cols = ["data_type", "run_id"]

        if os.path.exists(self.stats_path):
            stats_df = pd.read_csv(self.stats_path)
        else:
            stats_df = pd.DataFrame(columns=list(new_row.keys()))

        # Drop existing row with same key
        match = (
            (stats_df["data_type"] == new_row["data_type"]) &
            (stats_df["run_id"] == new_row["run_id"])
        )
        stats_df = stats_df[~match]

        # Append the new row
        stats_df = pd.concat([stats_df, pd.DataFrame([new_row])], ignore_index=True)

        # Save to CSV
        stats_df.to_csv(self.stats_path, index=False)



    def get_tree_structure_from_llm(self, cluster_groups, gfsts_names, use_presaved=False, batch_size=20, sleep_time=5):
        if use_presaved and os.path.exists(self.tree_data_name):
            with open(self.tree_data_name, "r") as f:
                return json.load(f)

        kv_actor = self.get_kv_store_actor()
        semantic_types_list = [[gfsts_names[i] for i in group] for group in cluster_groups]

        results = [None] * len(semantic_types_list)
        llm_tasks = []
        llm_indices = []

        for idx, semantic_types in enumerate(semantic_types_list):
            if len(semantic_types) == 1:
                results[idx] = {
                    "name": semantic_types[0],
                    "is_generated": False,
                    "children": []
                }
            else:
                llm_indices.append(idx)
                llm_tasks.append(semantic_types)

        for i in range(0, len(llm_tasks), batch_size):
            batch_semantic_types_list = llm_tasks[i:i + batch_size]
            batch_indices = llm_indices[i:i + batch_size]

            for j in range(len(batch_semantic_types_list)):
                prompt = cluster_to_tree_structure_prompt(batch_semantic_types_list[j])
                self.add_token_count(prompt, "tree_structure_call")

            batch_ray_tasks = [
                text_generation_func.remote(
                    f"{batch_indices[j]}/{len(semantic_types_list)}",
                    (batch_semantic_types_list[j],),
                    cluster_to_tree_structure_prompt,
                    self.llm_params,
                    kv_store_actor=kv_actor
                )
                for j in range(len(batch_semantic_types_list))
            ]

            batch_outputs = ray.get(batch_ray_tasks)

            for idx, output in zip(batch_indices, batch_outputs):
                try:
                    results[idx] = json.loads(output.strip("`"))
                except json.JSONDecodeError:
                    print(f"[Warning] Failed to parse LLM output:\n{output}")
                    results[idx] = None

            time.sleep(sleep_time)
        
        with open(self.tree_data_name, "w") as f:
            json.dump(results, f, indent=2)

        return results


    def build_forest(self, batch_size=1000, sleep_time=0, use_presaved=False):
        start_time = time.time()
        # Load graph
        g = load_clean_graph(self.data_type, self.run_id)
        # Get G-FSTs and cluster them
        gfsts_names, _, cluster_groups = get_and_cluster_gfsts(g, self.cluster_groups_name, self.embedding_cache_name, use_presaved=use_presaved)
        # Get tree structure from LLM
        semantic_tree_extraction = self.get_tree_structure_from_llm(cluster_groups, gfsts_names, use_presaved=use_presaved, batch_size=batch_size, sleep_time=sleep_time)
        # Build semantic forest
        g, root_node_names = build_semantic_forest(semantic_tree_extraction, gfsts_names, cluster_groups)
        save_semantic_forest(g, self.semantic_forest_name)
        end_time = time.time()
        self.log_stats(
            elapsed_time=end_time - start_time,
            num_trees=len(root_node_names),
            num_total_nodes=g.number_of_nodes()
        )
import os
import time
import json
import pickle
import hashlib
from collections import defaultdict

import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from graph_construction import load_clean_graph, get_nodes_by_node_type, NodeType
from ray_cmds import code_compiling_func, Params, KeyValueStore, text_generation_func
from prompts.prompt_cluster_to_tree_structure import cluster_to_tree_structure_prompt

import ray

def get_and_cluster_gfsts(g, cluster_groups_name, embedding_cache_name, use_presaved=False):
    """
    Load or compute G-FST names, mapping, and cluster groups. Uses cache if available.

    :param g: semantic graph
    :param cluster_groups_name: path to pickle file for saving/loading cache
    :param use_presaved: whether to load from cache
    :return: (gfsts_names, gfsts_name_to_node, cluster_groups)
    """
    if use_presaved and os.path.exists(cluster_groups_name):
        with open(cluster_groups_name, 'rb') as f:
            print(f"Loaded cluster data from {cluster_groups_name}")
            data = pickle.load(f)
            return data["gfsts_names"], data["gfsts_name_to_node"], data["cluster_groups"]

    # Compute from graph
    gfsts_nodes = get_nodes_by_node_type(g, NodeType.GENERAL_ENRICHED_SEMANTIC_TYPE)
    gfsts_name_to_node = {n.split(':')[-1]: n for n in gfsts_nodes}
    gfsts_names = list(gfsts_name_to_node.keys())

    print(f"[Clustering] Running DBSCAN on {len(gfsts_names)} G-FSTs...")
    cluster_groups = cluster_gfst(gfsts_names, embedding_cache_name)

    # Save to cache
    with open(cluster_groups_name, 'wb') as f:
        pickle.dump({
            "gfsts_names": gfsts_names,
            "gfsts_name_to_node": gfsts_name_to_node,
            "cluster_groups": cluster_groups
        }, f)
        print(f"Cluster info saved to {cluster_groups_name}")

    return gfsts_names, gfsts_name_to_node, cluster_groups

def build_semantic_forest(semantic_tree_extraction, gfsts_names, cluster_groups):
    """
    Construct a semantic forest graph from clustered tree structures.

    Guarantees:
      • All node names are unique
      • Naming-conflict rules:
          1. New generated node, name exists      → rename **new** node
          2. New non-generated, existing generated→ rename **existing** node
          3. New non-generated, existing non-gen  → rename **existing** node and mark it generated
      • Each root stores
          • semantic_type_list      – all *non-generated* semantic types added to the tree
          • semantic_type_generated – all *generated* semantic types added to the tree
    """
    graph = nx.DiGraph()
    root_node_names = []

    for cluster_index, tree in enumerate(semantic_tree_extraction):
        if tree is None:
            root_node_names.append(None)
            continue

        leaf_semantic_types = {gfsts_names[i] for i in cluster_groups[cluster_index]}
        non_generated_names = set()
        generated_names = set()
        actual_root_node_name = None

        def add_node_and_edges(node, parent_name=None):
            nonlocal actual_root_node_name
            raw_name = node["name"].lower()
            is_generated = node["is_generated"]

            if is_generated and raw_name in leaf_semantic_types:
                is_generated = False
            elif not is_generated and raw_name not in leaf_semantic_types:
                is_generated = True

            node_type = NodeType.SEMANTIC_TREE_ROOT if parent_name is None else NodeType.SEMANTIC_TREE_NODE
            prefix = "ROOT" if node_type == NodeType.SEMANTIC_TREE_ROOT else "TREE"
            base_node_name = f"{prefix}:{raw_name}"
            final_node_name = base_node_name
            current_parent_name = parent_name

            if base_node_name in graph:
                existing_is_generated = graph.nodes[base_node_name]["is_generated"]

                if is_generated:
                    i = 1
                    while f"{base_node_name}_{i}" in graph:
                        i += 1
                    final_node_name = f"{base_node_name}_{i}"
                    print(f"[Rename New Generated] {base_node_name} → {final_node_name}")

                elif not is_generated and existing_is_generated:
                    i = 1
                    while f"{base_node_name}_{i}" in graph:
                        i += 1
                    renamed_existing = f"{base_node_name}_{i}"
                    print(f"[Rename Old Generated] {base_node_name} → {renamed_existing}")
                    nx.relabel_nodes(graph, {base_node_name: renamed_existing}, copy=False)
                    graph.nodes[renamed_existing]["is_generated"] = True
                    if current_parent_name == base_node_name:
                        current_parent_name = renamed_existing
                    final_node_name = base_node_name

                else:
                    i = 1
                    while f"{base_node_name}_{i}" in graph:
                        i += 1
                    renamed_existing = f"{base_node_name}_{i}"
                    print(f"[Conflict Non-Generated] {base_node_name} → {renamed_existing}")
                    nx.relabel_nodes(graph, {base_node_name: renamed_existing}, copy=False)
                    graph.nodes[renamed_existing]["is_generated"] = True
                    if current_parent_name == base_node_name:
                        current_parent_name = renamed_existing
                    final_node_name = base_node_name
            if is_generated:
                generated_names.add(final_node_name.split(":")[-1])
            else:
                non_generated_names.add(raw_name)

            # Add node
            node_attrs = {
                "is_generated": is_generated,
                "node_type": node_type
            }
            if is_generated:
                node_attrs["semantic_meaning"] = node.get("semantic_meaning", "")
            if parent_name is None:
                actual_root_node_name = final_node_name

            graph.add_node(final_node_name, **node_attrs)

            if current_parent_name:
                graph.add_edge(current_parent_name, final_node_name)

            for child in node.get("children", []):
                add_node_and_edges(child, final_node_name)

        add_node_and_edges(tree)

        if actual_root_node_name:
            graph.nodes[actual_root_node_name]["non_generated"] = sorted(
                non_generated_names
            )
            graph.nodes[actual_root_node_name]["generated"] = sorted(
                generated_names
            )

        root_node_names.append(actual_root_node_name)

    return graph, root_node_names



def print_semantic_tree_json(node, level=0):
    """
    Recursively print semantic hierarchy tree showing only `name` and `is_generated`.
    
    :param node: dict, root of the tree or a subtree
    :param level: int, indentation level
    """
    indent = "  " * level
    print(f"{indent}- {node['name']} (generated: {node['is_generated']})")
    for child in node.get('children', []):
        print_semantic_tree_json(child, level + 1)

def print_semantic_tree(g, root, markerStr="+- ", levelMarkers=[]):
    emptyStr = " " * len(markerStr)
    connectionStr = "|" + emptyStr[:-1]
    level = len(levelMarkers)
    mapper = lambda draw: connectionStr if draw else emptyStr
    markers = "".join(map(mapper, levelMarkers[:-1]))
    markers += markerStr if level > 0 else ""

    is_generated = g.nodes[root].get('is_generated', 'N/A')
    print(f"{markers}{root} (generated: {is_generated})")

    children = list(g.successors(root))
    for i, child in enumerate(children):
        isLast = i == len(children) - 1
        print_semantic_tree(g, child, markerStr, [*levelMarkers, not isLast])




def get_embeddings(texts, model="text-embedding-3-small", dimensions=None, embedding_cache_name=None, batch_size=512):
    EMBEDDING_CACHE_PATH = embedding_cache_name

    def _hash_key(text, model, dimensions):
        key = f"{text}_{model}_{dimensions or 'default'}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    # Load or initialize cache
    if os.path.exists(EMBEDDING_CACHE_PATH):
        with open(EMBEDDING_CACHE_PATH, "rb") as f:
            EMBEDDING_CACHE = pickle.load(f)
    else:
        EMBEDDING_CACHE = {}

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Prepare queries
    embeddings = [None] * len(texts)
    to_query = []
    to_query_indices = []
    keys = []

    for i, text in enumerate(texts):
        key = _hash_key(text, model, dimensions)
        keys.append(key)
        if key in EMBEDDING_CACHE:
            embeddings[i] = EMBEDDING_CACHE[key]
        else:
            to_query.append((i, text, key))  # keep index + key together

    # Process in batches
    for i in range(0, len(to_query), batch_size):
        batch = to_query[i:i + batch_size]
        indices = [item[0] for item in batch]
        texts_batch = [item[1] for item in batch]
        batch_keys = [item[2] for item in batch]

        print(f"[Query] Fetching {len(texts_batch)} embeddings from OpenAI (batch {i // batch_size + 1})...")
        # Inside the batch loop, just before calling the API



        response = client.embeddings.create(input=texts_batch, model=model, dimensions=dimensions)
        new_embeddings = [np.array(res.embedding) for res in response.data]

        for idx, key, emb in zip(indices, batch_keys, new_embeddings):
            embeddings[idx] = emb
            EMBEDDING_CACHE[key] = emb

        # Save updated cache after each batch
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(EMBEDDING_CACHE, f)

    return np.vstack(embeddings)


def recursive_dbscan(embeddings, indices, gfsts_class_names, max_cols_per_cluster, eps_schedule, min_samples=1):
    """
    Recursively apply DBSCAN on a subset of indices with decreasing eps values.

    Args:
        embeddings (np.ndarray): full embedding array
        indices (list[int]): indices of current cluster
        gfsts_class_names (list[str]): all class names
        max_cols_per_cluster (int): max allowed per cluster
        eps_schedule (list[float]): decreasing eps values (e.g., [0.5, 0.3, 0.15])
        min_samples (int): min_samples for DBSCAN

    Returns:
        list[list[int]]: clustered indices
    """
    if not eps_schedule:
        return [indices]  # base case: return as single cluster

    eps = eps_schedule[0]
    current_embeddings = embeddings[indices]
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit_predict(current_embeddings)

    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1:
            cluster_map[label].append(indices[i])

    result_clusters = []
    for cluster_indices in cluster_map.values():
        if len(cluster_indices) > max_cols_per_cluster:

            # recursive split
            sub_clusters = recursive_dbscan(embeddings, cluster_indices, gfsts_class_names,
                                            max_cols_per_cluster, eps_schedule[1:], min_samples)
            result_clusters.extend(sub_clusters)
        else:
            result_clusters.append(cluster_indices)

    return result_clusters

def get_reduced_embeddings(embeddings, n_components=100):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

def cluster_gfst(gfsts_class_names, embedding_cache_name, max_cols_per_cluster=30, eps_schedule=list(np.arange(0.6, 0.1, -0.02)), min_samples=1):
    """
    Cluster G-FSTs by embedding and recursively split large clusters.

    Args:
        gfsts_class_names (list[str])
        max_cols_per_cluster (int)
        eps_schedule (list[float]): decreasing eps values for recursive DBSCAN
        min_samples (int)

    Returns:
        List[List[int]]: clusters
    """
    print(f"Cluster G-FSTs with {len(gfsts_class_names)} items")

    # Get embeddings
    all_embeddings = []
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    def get_reduced_embeddings(embeddings, n_components=100):
        max_components = min(embeddings.shape[0], embeddings.shape[1])
        n_components = min(n_components, max_components)
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)

    for chunk in chunked(gfsts_class_names, 512):
        emb = get_embeddings(texts=chunk, embedding_cache_name=embedding_cache_name)
        all_embeddings.append(emb)
    embeddings = np.vstack(all_embeddings)
    embeddings = get_reduced_embeddings(embeddings, n_components=100)

    # Run recursive DBSCAN
    clusters = recursive_dbscan(
        embeddings=embeddings,
        indices=list(range(len(gfsts_class_names))),
        gfsts_class_names=gfsts_class_names,
        max_cols_per_cluster=max_cols_per_cluster,
        eps_schedule=eps_schedule,
        min_samples=min_samples
    ) 
    return clusters

def load_semantic_forest(data_type, run_id):
    return pickle.load(open(f'assets/{data_type}/forest/{data_type}_semantic_forest{f"_{run_id}" if run_id else ""}/{data_type}_semantic_forest.pkl', "rb"))

def save_semantic_forest(semantic_forest, forest_name):
    pickle.dump(semantic_forest, open(forest_name, "wb"))

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
import numpy as np
import pandas as pd
import os
import re
import json
import difflib
from collections import defaultdict
from unidecode import unidecode
import tqdm
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

from semantic_type_base_classes import (
    gen_base_class_file, 
    BASE_CLASSES, 
    create_string_representation_of_imports_for_datasets, 
    create_string_representation_of_imports_for_general_types, 
    create_string_representation_of_imports_for_cross_type_cast
)
from graph_construction import NodeType, alone_context
from prompt_utils import fix_code
from util import df_reader, df_reader_v2

gen_base_class_file()
from semantic_type_base_classes_gen import *

def process_class_name(c_name, keep_underscore=False):
    """
    Basic String parsing on class name.

    :param c_name: string name of a generated class
    :return: new class name
    """
    if keep_underscore:
        new_name = new_name = re.sub(r'[^a-zA-Z0-9_]', '', c_name).lower()
    else:
        new_name = ''.join(ch for ch in c_name if ch.isalnum()).lower()
    new_name = re.sub(r'id$', 'identifier', new_name)
    new_name = re.sub(r'desc$', 'description', new_name)
    new_name = re.sub(r'pct$', 'percent', new_name)
    new_name = re.sub(r'percentage$', 'percent', new_name)
    return new_name

def extract_class_and_mapping_dicts_from_str(data_df, sem_type_string_col_name, keep_underscore=False):
    """
    Extract T-FSTs and the mapping from Col -> T-FST from LLM response, stored in a column called "sem_type_string_col_name"

    :param data_df: input dataframe
    :param sem_type_string_col_name: name of column containing LLM response
    :return: dataframe enriched with columns "passes_ast" - if the generated code compiles, "class_dict" - class_name -> T-FST, "mapping_dict" - Col -> T-FST name
    """
    import ast

    data_df.loc[:, 'passes_ast'] = False
    data_df.loc[:, 'class_dict'] = None
    data_df.loc[:, 'mapping_dict'] = None

    FORBIDDEN_CLASS_NAMES = {
        'datetime': 'datetimeclass',
        'round': 'roundclass',
        'yield': 'yieldclass'
    }
    for ix, row in data_df.loc[~data_df[sem_type_string_col_name].isna()].iterrows():
        sem_type_big_string = row[sem_type_string_col_name]
        module = ast.parse(sem_type_big_string)
        classes = {}
        imports = []
        other_inheritor = {}
        mapping = None

        forbidden_class_replace = True
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                node.name = process_class_name(node.name, keep_underscore=keep_underscore)
                if node.name in FORBIDDEN_CLASS_NAMES:
                    node.name = FORBIDDEN_CLASS_NAMES[node.name]
                    forbidden_class_replace = True

                for func_def in node.body:
                    if isinstance(func_def, ast.FunctionDef) and func_def.name == '__init__':
                        func_def.args.args.append(ast.arg(arg='*args', annotation=None))
                        func_def.args.args.append(ast.arg(arg='**kwargs', annotation=None))

                c_name = node.name
                c_body = ast.unparse(node)
                classes[c_name] = c_body

                if node.bases:
                    base_classes = [base.id for base in node.bases]
                    assert len(base_classes) == 1
                    base_class_name = base_classes[0]
                    if not base_class_name in BASE_CLASSES.keys():
                        base_class_name = process_class_name(base_class_name, keep_underscore=keep_underscore)
                        if base_class_name in FORBIDDEN_CLASS_NAMES:
                            base_class_name = FORBIDDEN_CLASS_NAMES[base_class_name]
                        other_inheritor[c_name] = base_class_name
            elif isinstance(node, ast.Assign):
                mappings = list(filter(lambda x: isinstance(x, ast.Name) and (x.id.lower() == 'mapping'), node.targets))
                if len(mappings) > 0:
                    mapping_node = node.value
                    if not isinstance(mapping_node, ast.Dict):
                        print(f"[Skip] Row {ix}: MAPPING is not a dict ({type(mapping_node).__name__})")
                        continue
                    pop_ixs = []

                    for idx in range(len(mapping_node.keys)):
                        if isinstance(mapping_node.keys[idx], ast.Constant):
                            mapping_node.keys[idx].value = mapping_node.keys[idx].value.strip(
                                ' ')  # sometimes the columns have spaces at the end, sometimes they dont. either way we gotta get rid of them.

                    for idx in range(len(mapping_node.values)):
                        val_node = mapping_node.values[idx]

                        if isinstance(val_node, ast.Constant):
                            pop_ixs.append(idx)
                            continue

                        if isinstance(val_node, ast.Call) and isinstance(val_node.func, ast.Name):
                            # Replace obj() → obj
                            val_node = ast.Name(id=val_node.func.id, ctx=ast.Load())
                            mapping_node.values[idx] = val_node

                        if isinstance(val_node, ast.Name):
                            val_id = val_node.id
                            if (val_id in FORBIDDEN_CLASS_NAMES) and forbidden_class_replace:
                                val_id = FORBIDDEN_CLASS_NAMES[val_id]
                            val_id = process_class_name(val_id, keep_underscore=keep_underscore)

                            if val_id not in classes:
                                pop_ixs.append(idx)
                            else:
                                mapping_node.values[idx].id = val_id  # apply cleaned name
                        else:
                            # e.g., Lambda, Attribute, etc. → skip
                            pop_ixs.append(idx)

                    keys = [mapping_node.keys[ix] for ix in range(len(mapping_node.values)) if ix not in pop_ixs]
                    values = [mapping_node.values[ix] for ix in range(len(mapping_node.values)) if ix not in pop_ixs]

                    mapping_node.keys = keys
                    mapping_node.values = values

                    mapping = ast.unparse(node.value)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                imports.append(ast.unparse(node))  # need to add imports to the top of the class

        if (mapping is None):
            print('Error w/ mapping: ', ix)

        if len(classes) == 0:
            print('Error w/ classes: ', ix)

        str_classes = {}
        for k, v in classes.items():
            str_classes[k] = create_string_representation_of_imports_for_datasets() + '\n' + '\n'.join(imports) + '\n'
            parent = other_inheritor.get(k)
            if parent and parent in classes:
                str_classes[k] += classes[parent] + '\n' + v
            else:
                str_classes[k] += v

        if len(classes) > 0 and (mapping is not None):
            data_df.at[ix, 'passes_ast'] = True
            data_df.at[ix, 'class_dict'] = str_classes
            data_df.at[ix, 'mapping_dict'] = mapping

def _run_per_col(ix, row, prefix):
    class_obj_map = {}
    col_mapping = {}

    try:
        # Compile all class definitions safely
        for class_name, string_class_def in row.class_dict.items():
            try:
                exec(string_class_def, globals())
                class_obj_map[class_name] = eval(f'{class_name}()')
            except Exception as e:
                print(f"[Skip Class] Failed to compile '{class_name}' in row {ix}: {e}")
                continue

        # Evaluate mapping dict
        try:
            col_mapping = eval(row.mapping_dict)
        except Exception as e:
            print(f"[Skip Table] Failed to eval mapping_dict in row {ix}: {e}")
            return []

        if 'harvard' in prefix:
            data_table_head = df_reader_v2(os.path.join(f"{prefix}/{row['data_product']}/{row['file_name']}"), max_rows=1e3)
        else:
            data_table_head = df_reader(os.path.join(f"{prefix}/{row['data_product']}/{row['file_name']}"), max_rows=1e3)
        kaggle_run_data = []

        for col_name, sem_type_class in col_mapping.items():
            if sem_type_class is None:
                print(f"[Skip Col] {col_name} is None")
                continue

            try:
                sem_type_class_name = sem_type_class.__name__
                if sem_type_class_name not in class_obj_map:
                    print(f"[Skip Col] {sem_type_class_name} not in compiled class_obj_map (row {ix})")
                    continue

                if col_name not in data_table_head.columns:
                    print(f"[Skip Col] {col_name} not found in original table {row['file_name']}")
                    continue

                col_values = data_table_head[col_name].values
                kaggle_run_data.append([
                    ix,
                    row['data_product'],
                    row.file_name,
                    col_name,
                    col_values,
                    sem_type_class_name,
                    row.class_dict[sem_type_class_name],
                    class_obj_map[sem_type_class_name],
                ])
            except Exception as e:
                print(f"[Skip Col] Error processing {col_name} in row {ix}: {e}")
                continue

        return kaggle_run_data

    except Exception as e:
        print(f"[Skip Row] Unhandled error in row {ix}: {e}")
        return []



def run_per_col(data_df, prefix):
    """
    Unrolls the input dataframe into a new dataframe, where before, each row corresponded to a data-table, but the result
    is a dataframe where each row corresponds to a T-FST.
    """
    kaggle_run_data = []
    for ix in tqdm.tqdm(data_df.loc[data_df.passes_ast].index):
        row = data_df.loc[ix]
        kag_row_data = _run_per_col(ix, row, prefix)
        kaggle_run_data += kag_row_data
    results_df = pd.DataFrame(kaggle_run_data,
                              columns=['df_ix', 'data_product', 'file_name', 'col_name', 'raw_col_values', 'class_name',
                                       'str_class_def', 'obj_class_def'])
    return results_df


def build_general_types(g, name_and_enriched_type_list, general_to_sub_type_map, force_gen_type_name=False):
    """
    Adds G-FSTs to graph given LLM response.

    :param g: networkx graph (tree at this point) containing Col -> T-FST -> P-FST
    :param name_and_enriched_type_list: list of lists, where each sub-list is a G-FST name, and its class (from LLM)
    :param general_to_sub_type_map: mapping from G-FST name to list of P-FST names
    :param force_gen_type_name: sometimes the G-FST class name doesn't match the one specified in general_to_sub_type_map. This boolean forces a renaming.
    :return:
    """
    import ast

    # human corrections
    correction = {
        'stocksereies': 'stockseries',
        'deliverblepercentage': 'deliverablepercentage',
        'stockpercentdeliverble': 'stockpercentdeliverable',
        'availablity': 'availability',
        'sociaeconomicstatus': 'socioeconomicstatus',
    }

    FORBIDDEN_CLASS_NAMES = {
        'round': 'roundclass',
    }

    d_to_create = defaultdict(set)
    for ix, (general_type_name, str_class_def) in enumerate(name_and_enriched_type_list):
        str_class_def = str_class_def.replace('@property', '')

        try:
            module = ast.parse(str_class_def)
        except SyntaxError as e:
            print(f"[Skip] Syntax error in {general_type_name}: {e}")
            continue

        classes = {}
        imports = []
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                if node.name in FORBIDDEN_CLASS_NAMES:
                    node.name = FORBIDDEN_CLASS_NAMES[node.name]

                if not force_gen_type_name:
                    c_name = node.name
                else:
                    node.name = general_type_name
                    c_name = general_type_name

                for func_def in node.body:
                    if isinstance(func_def, ast.FunctionDef) and func_def.name == '__init__':
                        func_def.args.args.append(ast.arg(arg='*args', annotation=None))
                        func_def.args.args.append(ast.arg(arg='**kwargs', annotation=None))
                try:
                    c_body = ast.unparse(node)
                except Exception as e:
                    print(f"[Skip] Cannot unparse AST node for {general_type_name}: {e}")
                    continue
                classes[c_name] = c_body
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                try:
                    imports.append(ast.unparse(node))  # need to add imports to the top of the class
                except Exception:
                    pass

        if general_type_name in correction:
            actual_class_name = correction[general_type_name]
        else:
            actual_class_name = general_type_name

        if actual_class_name not in classes:
            print(f"[Skip] Class {actual_class_name} not found in generated code for {general_type_name}")
            continue

        try:
            string_class_def = create_string_representation_of_imports_for_general_types() + '\n' + '\n'.join(imports) + '\n' + classes[actual_class_name]
            obj = alone_context(string_class_def, actual_class_name)
        except Exception as e:
            print(f"[Skip] Failed to compile or instantiate {actual_class_name}: {e}")
            continue

        dst = f'TYPE:_:_:{actual_class_name}'
        if dst not in g.nodes():
            g.add_node(
                dst,
                node_type=NodeType.GENERAL_ENRICHED_SEMANTIC_TYPE,
                str_class_def=string_class_def,
                obj_class_def=obj
            )

        for src in general_to_sub_type_map[general_type_name]:
            g.add_edge(src, dst)


def add_cross_type_casts(g, matches_per_gen_type, cross_type_cast_string_list, use_close_matches=False, keep_underscore=False):
    """
    Adds cross_type_casts to graph given LLM response.

    :param g: networkx graph (tree at this point) from Col -> T-FST -> P-FST -> G-FST
    :param matches_per_gen_type: mapping from G-FST name to other G-FST names that are close in vector space.
    :param cross_type_cast_string_list: LLM response containing cross_type_cast functions per G-FST name. (same length as matches_per_gen_type, and same order)
    :param use_close_matches: sometimes the generated names don't match the graph, so we use close matches
    """
    import ast
    import difflib

    remove_edges = []
    for e in g.edges():
        if 'cross_type_cast' in g.edges[e]:
            remove_edges.append(e)

    if len(remove_edges) > 0:
        print(f'Removing {len(remove_edges)} edges')
        g.remove_edges_from(remove_edges)

    forbidden_name = 'TYPE:_:_:agregaçãodaseleiçõesparecerista3'
    for gen_type, cross_type_cast_string in zip(matches_per_gen_type.keys(), cross_type_cast_string_list):
        if (gen_type in [forbidden_name, unidecode(forbidden_name)]) or (
                cross_type_cast_string in [np.nan, float('NaN'), None]):
            continue

        try:
            module = ast.parse(cross_type_cast_string.strip("\n`"))
        except Exception as e:
            print(f"[Skip] Syntax error in cross_type_cast for {gen_type}: {e}")
            continue

        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef):
                has_return = any(
                    isinstance(body, ast.Return) and not (
                        isinstance(body.value, ast.Constant) and body.value.value is None
                    )
                    for body in node.body
                )

                if has_return:
                    try:
                        relevant_substring = node.name[len('cross_type_cast_between_'):]
                        src, target = relevant_substring.split('_and_')
                        if keep_underscore:
                            src = src.strip()
                            target = target.strip()
                        else:
                            src = ''.join(src.split('_'))
                            target = ''.join(target.split('_'))
                        
                        node.name = f'cross_type_cast_between_{src}_and_{target}'

                        if ('id' in src) and ('id' in target):
                            continue

                        root_node = f'TYPE:_:_:{src}'
                        target_node = f'TYPE:_:_:{target}'

                        if (target_node not in g.nodes()) and use_close_matches:
                            candidates = matches_per_gen_type.get(root_node, [])
                            close = difflib.get_close_matches(target_node, candidates)
                            if close:
                                print(f'**Using {close[0]} instead of {target_node} ')
                                target_node = close[0]
                            else:
                                print(f"[Skip] Target node not found and no close match: {target_node}")
                                continue

                        if root_node not in g.nodes():
                            print(f"[Skip] Root node not found: {root_node}")
                            continue
                        if target_node not in g.nodes():
                            print(f"[Skip] Target node not found: {target_node}")
                            continue

                        g.add_edge(root_node, target_node)
                        g.edges[(root_node, target_node)]['cross_type_cast'] = (
                            create_string_representation_of_imports_for_cross_type_cast()
                            + '\n'
                            + ast.unparse(node)
                        )
                    except Exception as e:
                        print(f"[Skip] Unexpected error when adding edge for {gen_type}: {e}")
                        continue

def clean_and_standardize_class_defs(df, keep_underscore=True):
    import ast
    FORBIDDEN_CLASS_NAMES = {
        'datetime': 'datetimeclass',
        'round': 'roundclass',
        'yield': 'yieldclass'
    }

    all_enriched_defs = []
    updated_class_names = []

    for _, row in df.iterrows():
        raw_class_code = row['class_def']
        try:
            module = ast.parse(raw_class_code)
        except SyntaxError:
            all_enriched_defs.append(None)
            updated_class_names.append(None)
            continue

        classes = {}
        imports = []
        other_inheritor = {}

        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                original_name = node.name
                node.name = process_class_name(node.name, keep_underscore=keep_underscore)
                if node.name in FORBIDDEN_CLASS_NAMES:
                    node.name = FORBIDDEN_CLASS_NAMES[node.name]

                for func_def in node.body:
                    if isinstance(func_def, ast.FunctionDef) and func_def.name == '__init__':
                        func_def.args.args.append(ast.arg(arg='*args', annotation=None))
                        func_def.args.args.append(ast.arg(arg='**kwargs', annotation=None))

                class_name = node.name
                class_body = ast.unparse(node)
                classes[class_name] = class_body

                if node.bases:
                    base_classes = [base.id for base in node.bases if isinstance(base, ast.Name)]
                    if len(base_classes) == 1:
                        base_class_name = process_class_name(base_classes[0], keep_underscore=keep_underscore)
                        if base_class_name in FORBIDDEN_CLASS_NAMES:
                            base_class_name = FORBIDDEN_CLASS_NAMES[base_class_name]
                        other_inheritor[class_name] = base_class_name

            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                imports.append(ast.unparse(node))

        if len(classes) != 1:
            print(f"Error: {len(classes)} classes found (expected 1)")
            all_enriched_defs.append(None)
            updated_class_names.append(None)
            continue

        cls_name, cls_body = next(iter(classes.items()))
        enriched = create_string_representation_of_imports_for_datasets() + '\n' + '\n'.join(imports) + '\n'
        if cls_name in other_inheritor:
            parent_cls = other_inheritor[cls_name]
            if parent_cls in classes:
                enriched += classes[parent_cls] + '\n'
        enriched += cls_body

        all_enriched_defs.append(enriched)
        updated_class_names.append(cls_name)

    return all_enriched_defs, updated_class_names


def materialize_flattened_per_col(df, table_dir, keep_underscore=True):
    """
    Given a flattened dataframe with one row per column (and a `class_def` field),
    generate a new dataframe where each row corresponds to a fully-constructed T-FST,
    filtering out rows with invalid class definitions.
    """
    import ast

    kaggle_run_data = []

    for ix, row in df.iterrows():
        raw_code = row.get("class_def")
        if pd.isna(raw_code) or not isinstance(raw_code, str) or raw_code.strip() == "":
            print(f"[Skip] Row {ix} has no class_def.")
            continue

        try:
            module = ast.parse(raw_code)
            class_defs = [n for n in ast.walk(module) if isinstance(n, ast.ClassDef)]
            if len(class_defs) != 1:
                print(f"[Skip] Row {ix}: found {len(class_defs)} classes (expected 1).")
                continue
            class_name = process_class_name(class_defs[0].name, keep_underscore=keep_underscore)
        except Exception as e:
            print(f"[Skip] Row {ix} AST parsing failed: {e}")
            continue

        try:
            exec(raw_code, globals())
            class_obj = eval(f"{class_name}()")
        except Exception as e:
            print(f"[Skip] Row {ix} exec/eval failed: {e}")
            continue

        try:
            data_path = os.path.join(table_dir, row["data_product"], row["file_name"])
            table = df_reader(data_path, max_rows=1e3)

            if row["col_name"] not in table.columns:
                print(f"[Skip] Row {ix} col '{row['col_name']}' not in table.")
                continue

            raw_values = table[row["col_name"]].values

            kaggle_run_data.append([
                ix,
                row["data_product"],
                row["file_name"],
                row["col_name"],
                raw_values,
                class_name,
                raw_code,
                class_obj,
            ])
        except Exception as e:
            print(f"[Skip] Row {ix} failed during data loading: {e}")
            continue

    return pd.DataFrame(
        kaggle_run_data,
        columns=["df_ix", "data_product", "file_name", "col_name", "raw_col_values", "class_name", "str_class_def", "obj_class_def"]
    )

def extract_class_defs_via_mapping(data_df: pd.DataFrame, code_list: list[str]) -> pd.DataFrame:

    mapping_pattern = re.compile(r'MAPPING\s*=\s*\{(.*?)\}', re.DOTALL)
    class_pattern = re.compile(
        r'class\s+(\w+)\s*(\([^\)]*\))\s*:(.*?)(?=\nclass\s|\nMAPPING\s|\Z)',
        re.DOTALL
    )

    # Step 1: Build index: data_product → cumulative mapping and class pool
    code_by_product = {}

    for i, code in enumerate(code_list):
        mapping_match = mapping_pattern.search(code)
        if not mapping_match:
            continue

        mapping_str = mapping_match.group(1)
        mapping_dict = {}
        for line in mapping_str.strip().splitlines():
            if ':' not in line:
                continue
            k, v = line.strip().split(':', 1)
            k_clean = k.strip().strip("'\"").replace('.csv', '')
            v_clean = v.strip().strip(', ')
            mapping_dict[k_clean] = v_clean

        if mapping_dict:
            example_key = next(iter(mapping_dict.keys()))
            product = example_key.split('/')[0]
        else:
            continue

        # Parse classes
        class_pool = {
            cls_name.strip(): f"class {cls_name}{inherit.strip()}:{cls_body}"
            for cls_name, inherit, cls_body in class_pattern.findall(code)
        }

        # If product already exists, merge
        if product in code_by_product:
            code_by_product[product]['mapping'].update(mapping_dict)
            code_by_product[product]['class_pool'].update(class_pool)
        else:
            code_by_product[product] = {
                'mapping': mapping_dict,
                'class_pool': class_pool,
                'code_index': i
            }


    # Step 2: Match for each row in data_df
    class_def_list = []
    class_name_list = []
    code_index_list = []

    for _, row in data_df.iterrows():
        
        product = row['data_product']
        file_base = row['file_name'].replace('.csv', '')
        col_name = row['col_name']
        key = f"{product}/{file_base}/{col_name}"
        
        info = code_by_product.get(product)
        if info is None:
            class_def_list.append(None)
            class_name_list.append(None)
            code_index_list.append(None)
            continue

        class_name = info['mapping'].get(key)
        if not class_name:
            class_def_list.append(None)
            class_name_list.append(None)
            code_index_list.append(None)
            continue

        class_def = info['class_pool'].get(class_name)
        if not class_def:
            class_def_list.append(None)
            class_name_list.append(None)
            code_index_list.append(None)
            continue

        class_def_list.append(class_def)
        class_name_list.append(class_name)
        code_index_list.append(info['code_index'])

    data_df = data_df.copy()
    data_df['class_def'] = class_def_list
    data_df['class_name'] = class_name_list
    data_df['code_index'] = code_index_list
    return data_df

def parse_to_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform str_col_summary and table_semantic_meaning into a single row for each column,
    incorporating semantic_type and semantic_meaning extracted from the metadata JSON.

    If a row's `table_semantic_meaning` field is not a valid JSON, the whole row is skipped.
    """
    rows = []

    for idx, row in df.iterrows():
        data_product = row['data_product']
        file_name = row['file_name']
        str_col_summary = row['str_col_summary']
        metadata_json = row['table_semantic_meaning']

        # Try parsing JSON field; if invalid, skip the row
        if pd.isna(metadata_json) or not metadata_json.strip():
            continue

        try:
            metadata = json.loads(metadata_json)
            domain_knowledge = metadata.get('domain_knowledge', "")
            columns_meta = metadata.get('columns', {})
        except Exception as e:
            print(f"[Skip row {idx}] Invalid JSON: {e}")
            continue

        # Parse column-wise semantic data
        parts = str_col_summary.split('-col:')
        for part in parts:
            part = part.strip()
            if not part:
                continue

            lines = part.split('\n')
            col_name = lines[0].strip()
            col_summary = '\n'.join(lines[1:]).strip()

            col_metadata = columns_meta.get(col_name, {})
            semantic_type = col_metadata.get("semantic_type", "")
            semantic_meaning = col_metadata.get("semantic_meaning", "")

            rows.append({
                'data_product': data_product,
                'file_name': file_name,
                'col_name': col_name,
                'semantic_type': semantic_type,
                'semantic_meaning': semantic_meaning,
                'str_col_summary': col_summary,
                'domain_knowledge': domain_knowledge
            })

    return pd.DataFrame(rows)


def split_large_cluster_by_semantic_type(cluster_indices, sub_df, max_cols_per_prompt):
    """
    Split a large cluster into multiple smaller clusters by grouping same semantic types together.
    Ensures all columns with the same semantic_type stay in the same sub-cluster.
    """
    type_to_indices = defaultdict(list)
    for idx in cluster_indices:
        stype = sub_df.loc[idx, 'semantic_type']
        type_to_indices[stype].append(idx)

    split_clusters = []
    current_group = []
    current_count = 0

    for stype, indices in sorted(type_to_indices.items(), key=lambda x: -len(x[1])):  # sort largest type first
        if len(indices) > max_cols_per_prompt:
            # If even a single semantic type group is too big, put it alone
            split_clusters.append(indices)
            continue

        if current_count + len(indices) > max_cols_per_prompt:
            if current_group:
                split_clusters.append(current_group)
            current_group = indices.copy()
            current_count = len(indices)
        else:
            current_group.extend(indices)
            current_count += len(indices)

    if current_group:
        split_clusters.append(current_group)

    return split_clusters

def prepare_prompt_batches(df: pd.DataFrame, max_cols_per_prompt=50, eps=0.5, min_samples=1, semantic_type_only=True, threshold=0.6):
    """
    For each data_product, either directly batch all columns (if few) or cluster + group them.
    
    :param df: DataFrame with at least columns ['data_product', 'semantic_type', 'semantic_meaning']
    :param max_cols_per_prompt: max number of columns per LLM prompt
    :param eps: DBSCAN epsilon
    :param min_samples: DBSCAN min_samples
    :return: Dict[data_product -> List[List[df indices]]] — prompt groups per data_product
    """

    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    product_to_batches = {}
    threshold = int(max_cols_per_prompt * threshold)

    for dp in df['data_product'].unique():
        sub_df = df[df['data_product'] == dp]
        if len(sub_df) <= max_cols_per_prompt:
            # No clustering or grouping needed
            product_to_batches[dp] = [list(sub_df.index)]
        else:
            if semantic_type_only:
                texts = sub_df['semantic_type'].fillna('')
            else:
                texts = sub_df['semantic_type'].fillna('') + " " + sub_df['semantic_meaning'].fillna('')
            embeddings = model.encode(texts.tolist())
            labels = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit_predict(embeddings)

            cluster_map = defaultdict(list)
            for idx, label in zip(sub_df.index, labels):
                if label != -1:
                    cluster_map[label].append(idx)

            new_cluster_map = {}
            new_cluster_id = 0
            for cid, indices in cluster_map.items():
                if len(indices) > max_cols_per_prompt:
                    refined_clusters = split_large_cluster_by_semantic_type(indices, sub_df, max_cols_per_prompt)
                    for refined in refined_clusters:
                        new_cluster_map[new_cluster_id] = refined
                        new_cluster_id += 1
                else:
                    new_cluster_map[new_cluster_id] = indices
                    new_cluster_id += 1

            cluster_map = new_cluster_map
            
            sorted_clusters = sorted(cluster_map.values(), key=len, reverse=True)

            groups = []
            current_group = []
            current_count = 0
            for cluster in sorted_clusters:
                cluster_size = len(cluster)

                if cluster_size >= threshold:
                    # First flush any current group (small clusters)
                    if current_group:
                        groups.append(current_group)
                        current_group = []
                        current_count = 0

                    # Then put large cluster as its own group
                    groups.append(cluster)

                else:
                    # Try to merge small clusters
                    if current_count + cluster_size > max_cols_per_prompt:
                        groups.append(current_group)
                        current_group = cluster.copy()
                        current_count = cluster_size
                    else:
                        current_group.extend(cluster)
                        current_count += cluster_size

            if current_group:
                groups.append(current_group)

            product_to_batches[dp] = groups


    return product_to_batches

def get_input_data_for_grouped_semantic_types(data_df: pd.DataFrame, product_to_batches: dict) -> pd.DataFrame:
    """
    Build a DataFrame where each row corresponds to one batch of columns for one product.
    Adds:
        - data_product
        - batch_indices
        - combined_summary_meaning
        - domain_knowledge_str: single string with "Table <name>: <knowledge>" per line
    """
    # First, collect one domain_knowledge per (product, file_name)
    table_dom = {}
    for _, row in data_df.iterrows():
        dp, fn, dk = row.data_product, row.file_name, row.domain_knowledge
        # only take first non-null per table
        if pd.notna(dk) and (dp, fn) not in table_dom:
            table_dom[(dp, fn)] = dk

    rows = []
    for product, batches in product_to_batches.items():
        # build the multi-line domain_knowledge_str for *this* product
        lines = []
        for (dp, fn), dk in table_dom.items():
            if dp == product:
                lines.append(f"Table {fn}: {dk}")
        domain_knowledge_str = "\n".join(lines)

        for batch in batches:
            sub = data_df.loc[batch]
            combined = []
            for _, crow in sub.iterrows():
                block = (
                    f"Table: {crow.file_name}\n"
                    f"-col: {crow.col_name}\n"
                    # f"*potential_semantic_type: {crow.semantic_type}\n"
                    f"*semantic_meaning: {crow.semantic_meaning}\n"
                    f"*statistical_summary: {crow.str_col_summary}"
                )
                combined.append(block)
            combined_summary_meaning = "\n\n".join(combined)

            rows.append({
                'data_product': product,
                'batch_indices': batch,
                'combined_summary_meaning': combined_summary_meaning,
                'domain_knowledge_str': domain_knowledge_str
            })

    return pd.DataFrame(rows)
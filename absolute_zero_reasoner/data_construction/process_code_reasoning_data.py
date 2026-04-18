from pathlib import Path
import argparse
import re

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from absolute_zero_reasoner.rewards.code_reward import format_python_code
from absolute_zero_reasoner.data_construction.prompts import get_code_problem_predictor_prompt
from absolute_zero_reasoner.data_construction.process_data import (
    chat_think_system,
    instruction_following,
)


def _build_prompt(extraction_type: str, raw_prompt: str):
    """Wrap the raw predictor prompt for the given extraction type."""
    if extraction_type == 'chat_think':
        return [
            {"role": "system", "content": chat_think_system},
            {"role": "user",   "content": raw_prompt},
        ]
    return [{"role": "user", "content": instruction_following.format(raw_prompt)}]

def process_livecodebench_execution(row):
    # Extract all function names from the code
    program_name_matches = re.findall(r'def\s+(\w+)\s*\(', row['problem'])
    if not program_name_matches:
        raise ValueError("Could not find any function names in code")

    # Extract the function name from the input
    input_match = re.search(r'(\w+)\(', row['input'])
    if not input_match:
        raise ValueError("Could not find function name in input")

    input_function_name = input_match.group(1)

    # Check if the function name from input appears in any of the defined functions
    if input_function_name not in program_name_matches:
        raise ValueError(f"Function '{input_function_name}' from input not found in code. Available functions: {program_name_matches}")

    # Use the function name from input for replacement
    program_name = input_function_name

    # Replace the program name with `f` in the code
    row['problem'] = re.sub(r'def\s+' + re.escape(program_name) + r'\s*\(', 'def f(', row['problem'])

    # Process the input: remove the function name and keep only the parameters
    row['input'] = re.sub(r'^\w+\s*\(|\)$', '', row['input']).strip()

    return row


def add_imports(problem):
    # Add necessary imports based on the content of the problem
    if 'collections' in problem:
        problem = 'import collections\n' + problem
    if 'Counter' in problem:
        problem = 'from collections import Counter\n' + problem
    if 'gcd' in problem:
        problem = 'from math import gcd\n' + problem
    if 'deque' in problem:
        problem = 'from collections import deque\n' + problem
    if '@cache' in problem:
        problem = 'from functools import cache\n' + problem
    if '= inf' in problem or '[inf]' in problem or 'inf)' in problem:
        problem = 'from math import inf\n' + problem
    if 'accumulate' in problem:
        problem = 'from itertools import accumulate\n' + problem
    if '@lru_cache' in problem:
        problem = 'from functools import lru_cache\n' + problem
    if 'defaultdict' in problem:
        problem = 'from collections import defaultdict\n' + problem
    if 'bisect' in problem:
        problem = 'import bisect\n' + problem
    if 'islice' in problem:
        problem = 'from itertools import islice\n' + problem
    if 'math.inf' in problem:
        problem = 'import math\n' + problem
    if 'prod(' in problem:
        problem = 'from math import prod\n' + problem
    if 'heapify(' in problem:
        problem = 'from heapq import heapify, heappop, heappush\n' + problem
    if 'reduce(' in problem:
        problem = 'from functools import reduce\n' + problem
    if 'comb(' in problem:
        problem = 'from math import comb\n' + problem
    problem = problem.replace('List', 'list').replace('Dict', 'dict').replace('Tuple', 'tuple').replace('Set', 'set')
    problem = problem.replace('from typing import list', 'from typing import List')
    return problem


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=-1)
    parser.add_argument('--extraction_type', choices=['answer', 'chat_think'], default='answer',
                        help="'answer' builds test_answer.parquet (base-model format); "
                             "'chat_think' builds test_chat_think.parquet (instruct-model format)")
    args = parser.parse_args()

    # 283, 452, 510
    ds = load_dataset('cruxeval-org/cruxeval')['test']
    ds = ds.map(lambda x: {'problem': format_python_code(x['code'])})
    output_data = []
    for i, data in enumerate(tqdm(ds, desc="Processing CruxEval")):
        prompt = get_code_problem_predictor_prompt('code_i', data['problem'], data['input'], data['output'])
        output_data.append({
            "data_source": 'cruxeval_i',
            "prompt": _build_prompt(args.extraction_type, prompt),
            "problem": data['problem'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": data['output']
            },
            "extra_info": {
                'split': 'test',
                'index': i,
                'metric': 'pred_code_i',
                'problem_type': 'code_i',
                'input': data['input'],
                'output': data['output'],
            }
        })
        prompt = get_code_problem_predictor_prompt('code_o', data['problem'], data['input'], data['output'])
        output_data.append({
            "data_source": 'cruxeval_o',
            "prompt": _build_prompt(args.extraction_type, prompt),
            "problem": data['problem'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": data['output']
            },
            "extra_info": {
                'split': 'test',
                'index': i + len(data),
                'metric': 'pred_code_o',
                'problem_type': 'code_o',
                'input': data['input'],
                'output': data['output'],
            }
        })

    # another ds:
    ds = load_dataset('livecodebench/execution')['test']
    ds = ds.map(lambda x: {'problem': format_python_code(x['code'])})
    ds = ds.remove_columns(['code'])
    ds = ds.map(process_livecodebench_execution)
    # normalize the code
    ds = ds.map(lambda x: {'problem': add_imports(x['problem'])})
    for i, data in enumerate(tqdm(ds, desc="Processing LiveCodeBench")):
        prompt = get_code_problem_predictor_prompt('code_i', data['problem'], data['input'], data['output'])
        output_data.append({
            "data_source": 'livecodebench',
            "prompt": _build_prompt(args.extraction_type, prompt),
            "problem": data['problem'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": data['output']
            },
            "extra_info": {
                'split': 'test',
                'index': i + len(data),
                'metric': 'pred_code_i',
                'problem_type': 'code_i',
                'input': data['input'],
                'output': data['output'],
            }
        })

    df = pd.DataFrame(output_data)
    if args.max_length > 0:
        df = df.iloc[:args.max_length]
    path = Path('data/code_reason')
    path.mkdir(parents=True, exist_ok=True)
    name = 'test_answer' if args.extraction_type == 'answer' else f'test_{args.extraction_type}'
    suffix = f'_{args.max_length}' if args.max_length > 0 else ''
    df.to_parquet(path / f'{name}{suffix}.parquet')

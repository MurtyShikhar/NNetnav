"""
    postprocess trajectories to add retroactive reasoning and stop action
"""

import argparse
import os
import json
import logging
import random
import time
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup

from agent import InstructionGenerator
from dataclasses import dataclass

from agent.prompts import *
import browsergym.miniwob  # register miniwob tasks as gym environments
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from nnetnav_utils import make_dask_client


LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def parse_action(output):
    action_splitter = "```"
    pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
    match = re.search(pattern, output)
    if match:
        return match.group(1).strip()
    else:
        return None


def config():
    parser = argparse.ArgumentParser(
        description="Run Postprocessing for nnetnav trajectories"
    )
    parser.add_argument("--data_dir", type=str, default="")

    # agent config
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument("--agent_type", type=str, default="prompt")
    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    parser.add_argument(
        "--environment_type",
        type=str,
        default="webarena",
        choices=["webarena", "miniwob"],
    )
    parser.add_argument(
        "--script_mode",
        type=str,
        default="all",
        choices=["add_stop_action", "retroactive_reasoning", "all"],
    )
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()
    return args


def get_retroactive_reasoner(args):
    if args.environment_type == "webarena":
        prompt_folder = "src/agent/prompts/jsons"
    elif args.environment_type == "miniwob":
        prompt_folder = "src/agent/prompts/jsons_miniwob"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")

    prompt = f"{prompt_folder}/p_retroactive_reasoning.json"

    llm_config = lm_config.construct_llm_config(args)
    with open(prompt, "r") as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    model = InstructionGenerator(
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
    )
    return model


def get_add_stop_action(args):
    if args.environment_type == "webarena":
        prompt = "src/agent/prompts/jsons/p_add_stop_action.json"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    llm_config = lm_config.construct_llm_config(args)
    with open(prompt, "r") as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    model = InstructionGenerator(
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
    )
    return model


@dataclass
class ReasoningFunc:
    all_actions: list
    all_observation: list
    instruction: str
    data_idx: int

    agent: InstructionGenerator

    def run(self):
        reasoner_agent = self.agent
        actions = self.all_actions
        states = self.all_observation
        instruction = self.instruction
        retroactive_reasoning = []
        for idx, state in enumerate(states):
            orig_action = actions[idx]
            action = orig_action
            output = reasoner_agent.generate(
                {
                    "action": action,
                    "state": state,
                    "instruction": instruction,
                },
                None,
            )
            retroactive_reasoning.append(output)
        return {"data_idx": self.data_idx, "reasoning": retroactive_reasoning}


@dataclass
class StopActionFunc:
    last_observation: str
    instruction: str
    data_idx: int
    agent: InstructionGenerator

    def run(self):
        data_idx = self.data_idx
        stop_action_agent = self.agent
        last_observation = self.last_observation
        instruction = self.instruction
        stop_output = stop_action_agent.generate(
            {"state": last_observation, "instruction": instruction}, None
        )
        return {"data_idx": data_idx, "stop_action": stop_output}


def run_reasoning(args):
    reasoner_agent = get_retroactive_reasoner(args)

    orig_data = json.load(open(os.path.join(args.data_dir, "filtered_parsed.json")))

    all_reasoning_funcs = []
    all_data = []
    for i, data in tqdm(enumerate(orig_data), total=len(orig_data)):
        actions = data["parsed_actions"]
        states = [m["user"] for m in data["messages"] if "user" in m]
        instruction = data["intent"]
        reasoning_func = ReasoningFunc(actions, states, instruction, i, reasoner_agent)
        all_data.append(data)
        all_reasoning_funcs.append(reasoning_func)

    if args.n_jobs == 1:
        all_reasoning_outputs = []
        for reasoning_func in all_reasoning_funcs:
            all_reasoning_outputs.append(reasoning_func.run())
    else:
        with ProgressBar():
            delayed_results = []
            for reasoning_func in all_reasoning_funcs:
                delayed_results.append(delayed(reasoning_func.run)())
            all_reasoning_outputs = compute(*delayed_results)

    for rout in all_reasoning_outputs:
        # might get jumbled up by dask so we need to index correctly
        all_data[rout["data_idx"]]["retroactive_reasoning"] = rout["reasoning"]

    out_data_path = os.path.join(
        args.data_dir, "filtered_parsed_with_retroactive_reasoning.json"
    )
    with open(out_data_path, "w") as f:
        json.dump(all_data, f)


def run_stop_action(args):
    """
    WebArena requires models to output a stop action.
    We postprocess nnetnav trajectories to add stop action.
    """
    stop_action_agent = get_add_stop_action(args)
    orig_data = json.load(
        open(
            os.path.join(
                args.data_dir, "filtered_parsed_with_retroactive_reasoning.json"
            )
        )
    )
    all_outputs = []
    all_stop_action_funcs = []
    all_data = []
    for i, data in tqdm(enumerate(orig_data), total=len(orig_data)):
        states = [m["user"] for m in data["messages"] if "user" in m]
        last_observation = states[-1]
        instruction = data["intent"]
        stop_action_func = StopActionFunc(
            last_observation, instruction, i, stop_action_agent
        )
        all_stop_action_funcs.append(stop_action_func)
        all_data.append(data)
    if args.n_jobs == 1:
        all_stop_outputs = []
        for stop_action_func in all_stop_action_funcs:
            all_outputs.append(stop_action_func.run())
    else:
        with ProgressBar():
            delayed_results = []
            for stop_action_func in all_stop_action_funcs:
                delayed_results.append(delayed(stop_action_func.run)())
            all_stop_outputs = compute(*delayed_results)

    for stop_output in all_stop_outputs:
        # might get jumbled up by dask so we need to index correctly
        all_data[stop_output["data_idx"]]["stop_action"] = stop_output["stop_action"]

    out_data_path = os.path.join(
        args.data_dir, "filtered_parsed_with_retroactive_stop_action.json"
    )
    with open(out_data_path, "w") as f:
        json.dump(all_data, f)


if __name__ == "__main__":
    args = config()
    if args.script_mode == "all":
        run_reasoning(args)
        run_stop_action(args)
    elif args.script_mode == "retroactive_reasoning":
        run_reasoning(args)
    else:
        assert os.path.exists(
            "{}/filtered_parsed_with_retroactive_reasoning.json".format(args.data_dir)
        ), "Retroactive reasoning must be run before stop action"
        run_stop_action(args)

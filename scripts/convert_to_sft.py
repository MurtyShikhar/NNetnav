# Code to convert web-agent trajectories into instruction tuning dataimport argparse
import argparse
import json, subprocess, tempfile, os
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.actions import create_id_based_action
import glob
from browser_env import ScriptBrowserEnv

from bs4 import BeautifulSoup
from llms.tokenizers import Tokenizer
import jsonlines

from browser_env.env_config import URL_MAPPINGS
from agentlab.llm.llm_utils import count_tokens


system_chat_message_webarena = {
    "role": "system",
    "content": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere's the information you'll have:\nThe user's objective: This is the task you're trying to complete.\nThe current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\nThe current web page's URL: This is the page you're currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous actions: These are all the action you have performed. It may be helpful to track your progress.\n\nThe actions you can perform fall into several categories:\n\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\nHomepage:\nIf you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation\n2. You should only issue one action at a time.\n3. You should follow the examples to reason step by step and then issue the next action.\n4. You are strictly forbidden from issuing a goto action to a URL that is not on the homepage.\n5. Generate the action in the correct format. Start by reasoning about the current situation. End with \"In summary, the next action I will perform is\" phrase, followed by action inside ``````. For example, \"Let's think step-by-step. Given the current state, I need to click on the like button which has id 1234. In summary, the next action I will perform is ```click [1234]```\".\n6. Issue stop action when you think you have achieved the objective. Don't generate anything after stop. \n\nHere are some example outputs for some random tasks:\n1. Let's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```\n2. Let's think step-by-step. This page has a search box whose ID is [164]. According to the nominatim rule of openstreetmap, I can search for the restaurants near a location by \"restaurants near\". I can submit my typing by pressing the Enter afterwards. In summary, the next action I will perform is ```type [164] [restaurants near CMU] [1]```",
}


def map_url_to_real(url: str) -> str:
    """Map the urls to their real world counterparts, for webarena"""
    URL_MAPPINGS = {
        ":9999": "http://reddit.com",
        ":7770": "http://onestopmarket.com",
        ":7780/admin": "http://luma.com/admin",
        ":8023": "http://gitlab.com",
        "User:The_other_Kiwix_guy/Landing": "http://wikipedia.org",
        ":3000": "http://openstreetmap.org",
        ":4399": "http://homepage.com",
    }

    for i, j in URL_MAPPINGS.items():
        # if the url is like http://<>:port/<something> then replace with real/<something>
        import re

        pattern = re.compile(r"http://.*" + i)
        match_obj = pattern.match(url)
        if match_obj:
            matched_stuff = match_obj.group(0)
            url = url.replace(matched_stuff, j)
    return url


def get_instruction_tuning_example(
    instruction_template, intent, observation, previous_action, message, url=None
):
    """
    convert intent, metadata, previous action and message into instruction tuning example
    """
    to_fill = {
        "observation": observation,
        "objective": intent,
        "past_actions": previous_action,
        "url": url,
    }
    user_instruction = instruction_template.format(**to_fill)
    assistant_output = message
    return [
        {"role": "user", "content": user_instruction},
        {"role": "assistant", "content": assistant_output},
    ]


def main(args, prompt_path):
    # not all environments need a stop action
    instruction_template = json.load(open(prompt_path, "r"))["template"]
    if os.path.exists(
        "{}/filtered_parsed_with_retroactive_stop_action.json".format(
            args.nnetnav_dem_dir
        )
    ):
        with open(
            "{}/filtered_parsed_with_retroactive_stop_action.json".format(
                args.nnetnav_dem_dir
            ),
            "r",
        ) as f:
            demonstrations = json.load(f)
    else:
        raise ValueError("Bad input.")
    all_instruction_tuning_examples = []
    for demonstration in demonstrations:
        observations = []
        actions = []
        dem_id = demonstration["task_id"]
        for m in demonstration["messages"]:
            if "user" in m:
                observations.append(m["user"])
            else:
                actions.append(m["assistant"])

        actions = demonstration["retroactive_reasoning"]
        # replace the last action with the stop action. Please see _convert_to_shorter_trajectory in run_lgs.py for context
        # for why this is the correct thing to do
        actions = actions[:-1] + [demonstration["stop_action"]]
        dem_size = len(observations) - 1
        with open(
            "{}/render_states/render_{}.html".format(args.nnetnav_dem_dir, dem_id), "r"
        ) as f:
            render_state = f.read()
            soup = BeautifulSoup(render_state, "html.parser")
            previous_actions = [
                obv.get_text() for obv in soup.find_all("div", {"class": "prev_action"})
            ]
        print(demonstration["intent"])
        previous_actions_curr = []
        for idx, obs in enumerate(observations):
            webpage = obs.split("observation:")[-1].strip()
            url = obs.split("observation:")[0].strip().split("URL:")[-1].strip()
            url = map_url_to_real(url)
            previous_actions_curr.append(previous_actions[idx])
            _with_steps = [
                "{}: {}".format(jdx + 1, a)
                for jdx, a in enumerate(previous_actions_curr)
            ]
            instruction_tune_example = get_instruction_tuning_example(
                instruction_template,
                demonstration["intent"],
                webpage,
                "\n".join(_with_steps),
                actions[idx],
                url,
            )
            chat_message = [system_chat_message_webarena] + instruction_tune_example
            # need to follow the following format for instruction tuning of llama
            full_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{}\n<|eot_id|>".format(
                chat_message[0]["content"],
                chat_message[1]["content"],
            )

            output = (
                "<|start_header_id|>assistant<|end_header_id|>\n{}\n<|eot_id|>".format(
                    chat_message[-1]["content"]
                )
            )

            task_name = "{}_{}".format(
                args.nnetnav_dem_dir.split("/")[-1], demonstration["task_id"]
            )
            n_tokens = count_tokens(full_prompt, args.model_name)
            if n_tokens > 100000:
                # too big a context for training
                continue
            output_curr = {
                "dataset": f"webarena_{args.exp_name}",
                "id": task_name,
                "output": output,
                "task_name": task_name,
                "prompt": full_prompt,
                "n_tokens": n_tokens,
                "messages": chat_message,
            }

            all_instruction_tuning_examples.append(output_curr)

    return all_instruction_tuning_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get Playwright traces for nnetnav demonstrations"
    )
    parser.add_argument(
        "--nnetnav_dem_dir",
        type=str,
        help="Directory where parsed nnetnav demonstrations are stored",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="Model name for token count",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="llm",
        help="Experiment name for tracking",
    )

    args = parser.parse_args()
    prompt_path = "src/agent/prompts/jsons/p_cot_llama_action_history.json"

    outputs = main(args, prompt_path)
    all_task_names = [o["task_name"] for o in outputs]
    print("Created {} supervised finetuning examples".format(len(outputs)))

    train_size = int(0.9 * len(all_task_names))
    val_size = int(0.1 * len(all_task_names))
    train_tasks = set(all_task_names[:train_size])
    val_tasks = set(all_task_names[train_size:])

    train_outputs = [o for o in outputs if o["task_name"] in train_tasks]
    val_outputs = [o for o in outputs if o["task_name"] in val_tasks]
    test_outputs = val_outputs
    DATA_DUMP_DIR = "/u/scr/smurty/agents-with-exploration/public/nnetnav_datasets"
    os.makedirs(f"{DATA_DUMP_DIR}/{args.exp_name}", exist_ok=True)

    with open(
        f"{DATA_DUMP_DIR}/{args.exp_name}/train.jsonl", "w", encoding="utf-8"
    ) as f:
        for item in train_outputs:
            f.write(json.dumps(item) + "\n")

    with open(f"{DATA_DUMP_DIR}/{args.exp_name}/val.jsonl", "w", encoding="utf-8") as f:
        for item in val_outputs:
            f.write(json.dumps(item) + "\n")

    with open(
        f"{DATA_DUMP_DIR}/{args.exp_name}/test.jsonl", "w", encoding="utf-8"
    ) as f:
        for item in test_outputs:
            f.write(json.dumps(item) + "\n")

    # take the val and test tasks and write them to eval_tasks.txt
    with open(f"{DATA_DUMP_DIR}/{args.exp_name}/eval_tasks.txt", "w") as f:
        for task in val_tasks:
            f.write(task + "\n")

    # also write the full data
    with open(f"{DATA_DUMP_DIR}/{args.exp_name}/all.jsonl", "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")

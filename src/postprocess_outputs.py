"""
    a quick (and hacky) postprocessing script for nnetnav trajectories to add parsed actions, and move around browser states + HTML renderings
"""

import argparse
import os, json
import glob
import copy
from agent.prompts import *


def set_paths():
    """
    If we do not do this, then webarena can complain.
    Does not even matter what the paths are, as long as they are set.
    TODO(smurty): fix this hack later
    """
    os.environ["GITLAB"] = "PASS"
    os.environ["SHOPPING"] = "PASS"
    os.environ["SHOPPING_ADMIN"] = "PASS"
    os.environ["REDDIT"] = "PASS"
    os.environ["GITLAB"] = "PASS"
    os.environ["MAP"] = "PASS"
    os.environ["WIKIPEDIA"] = "PASS"
    os.environ["HOMEPAGE"] = "PASS"
    return


def parse_trajectories(trajectories_all, filter_dir):
    """
    Parse trajectories, so that we can get actual low-level actions from LM generated strings
    """

    ipath = "src/agent/prompts/jsons/p_cot_id_actree_2s.json"
    lm_config = None
    tokenizer = None
    prompt = CoTPromptConstructor(ipath, lm_config, tokenizer)

    trajectories_parsed = []
    for t in trajectories_all:
        t_so_far = []
        for m in t["messages"]:
            if "user" in m:
                continue
            else:
                try:
                    parsed_response = prompt.extract_action(m["assistant"])
                    t_so_far.append(parsed_response)
                except:
                    action_splitter = "`"
                    pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
                    match = re.search(pattern, m["assistant"])
                    if match and match.group(1).strip():
                        t_so_far.append(match.group(1).strip())
                    else:
                        print("Error")
                        t_so_far.append(m["assistant"])
                    continue
        curr_trajectory = copy.deepcopy(t)
        # print(t_so_far)
        curr_trajectory["parsed_actions"] = t_so_far
        trajectories_parsed.append(curr_trajectory)

    print(
        "Found {} trajectories after prefix relabeling".format(len(trajectories_parsed))
    )
    with open("{}/filtered_parsed.json".format(filter_dir), "w") as f:
        json.dump(trajectories_parsed, f)


def postprocess_outputs(orig_dir, filter_dir):

    all_renderings = [
        render_state for render_state in glob.glob(f"{orig_dir}/render_*.html")
    ]
    all_renderings_remapped = {}
    for rendering in all_renderings:
        task_id = int(os.path.basename(rendering).split("_")[-1].split(".")[0])
        all_renderings_remapped[task_id] = rendering

    trajectories_all = json.load(
        open(os.path.join(filter_dir, "filtered_demonstrations_0.json"))
    )
    browser_states_all = json.load(open(f"{orig_dir}/test.raw.json"))
    browser_states_remapped = {}
    for state in browser_states_all:
        browser_states_remapped[state["task_id"]] = state

    browser_states = []
    render_states = []
    for traj in trajectories_all:
        task_id = traj["task_id"]
        print(traj["intent"])
        browser_states.append(browser_states_remapped[task_id])
        render_states.append(all_renderings_remapped[task_id])

    if not os.path.exists(f"{filter_dir}/browser_states"):
        os.makedirs(f"{filter_dir}/browser_states")

    if not os.path.exists(f"{filter_dir}/render_states"):
        os.makedirs(f"{filter_dir}/render_states")

    for i, state in enumerate(browser_states):
        state_id = state["task_id"]
        with open(f"{filter_dir}/browser_states/config_{state_id}.json", "w") as f:
            json.dump(state, f)

    for i, state in enumerate(render_states):
        print(state)
        os.system(f"cp {state} {filter_dir}/render_states/")

    parse_trajectories(trajectories_all, filter_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_dir", type=str, default="")
    parser.add_argument("--filter_dir", type=str, default="")
    args = parser.parse_args()

    orig_dir = args.orig_dir
    filter_dir = args.filter_dir

    postprocess_outputs(orig_dir, filter_dir)

"""
Runs a vision based reward model.
"""

import base64
from PIL import Image
from agentlab.agents.visualwebarena.agent import image_data_to_uri
import numpy as np
import pickle, gzip
import glob
import requests
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import re


SYSTEM_PROMPT = """
An autonomous intelligent agent navigating a web browser is given an instruction by a user. Your objective is to give a score to the agent based on how well it completed its task. Your score must be on the scale of 1 to 5. Give a score of 5 only when there are no errors. To do this task you are provided with the following information:

## Objective: This is the task the agent is given.
## Trajectory: A sequence of the form state_0, action_0, state_1, action_1, state_2, ... action_(T), denoting all interactions of the agent. In particular, state_i is the screenshot at time-step i, and action_i is the action taken at that step. We describe the actions below:

The actions of the agent come from the following action set, that allows the agent to interact with the browser. The primary way of referring to elements in the page is through pixel coordinates (x, y) corresponding to pixel positions on the screenshot, where x is pixel from the left edge and y is pixel from the top edge.

Page Operation Actions:
- `click (x,y)`: Clicks at the coordinate (x, y) on the current page.
- `dblclick (x,y)`: Double click at the coordinate (x,y) on the current page.
- `type (x,y) [text]`: Clicks the element at (x, y), then types the given text, and presses Enter by default. If there is already some text on the textbox element at (x,y), it will be cleared before typing in the new text.
- `type (x,y) [text] [press_enter_after=0|1]`: Optional flag to control whether "Enter" is pressed after typing (default is 1). As before, if there is already some text at the textbox in (x,y) it will be cleared before the new input is typed.
- `type_no_clear (x,y) [chars]`: types in chars at the textbox at position (x, y), but doesn't clear text if present at (x,y). Useful for prompting autocomplete, and type in a few characters first, see autocomplete results, and then continue typing to narrow down search. For this action, "Enter" is not pressed.
- `hover (x,y)`: Moves the cursor to hover over the point (x, y).
- `press [key_comb]`: Simulates a keyboard action like Ctrl+v or Backspace (to delete a character) or Meta+A (to select text in currently focused element) or Enter (to submit something).
- `scroll [down|up]`: Scrolls the full page up or down.

Tab Management Actions:
- `new_tab`: Opens a new, empty browser tab.
- `tab_focus [tab_index]`: Switches to a specific tab using its index.
- `close_tab`: Closes the currently active tab.

URL Navigation Actions:
- `goto [url]`: Navigates to a specific URL.
- `go_back`: Goes back to the previous page.
- `go_forward`: Moves forward if a previous go_back was issued.

Completion Action:
- `stop [answer]`: Used when the task is complete. If a textual answer is needed, include it inside the brackets. If you believe the task is impossible, respond with stop [N/A].

Homepage: To visit other websites, go to the homepage: http://homepage.com. It contains a list of allowed websites.
---
Your answer should follow the following format strictly:
[thought]
Use this space to think about the task and how well it was executed. Consider the trajectory of actions taken by the agent, and whether they align with the objective given. Think about whether the agent followed the steps needed to complete the task correctly, and if there were any errors in execution.
[thought]

[reward]
<your-answer from 1 to 5, based on the scoring guidelines below>
[reward]
---
Here are some guidelines for scoring:
1. Give a score of 5 if there are no errors.
2. Give a score of 4 if the task was almost correctly done (i.e. for form filling, most of the fields are filled or for a search task, a query was correctly typed, and the agent navigated to the right links).
3. Give a score of 3 if the task was only partially completed (i.e for form filling, less than half the fields are filled out) and if there are other minor execution errors.
4. Give a score of 1 or 2 if there are major execution errors, or the task was hardly completed, or if the agent did something completely unrelated.
---
To be successful, it is very important to follow the following rules:
1. Explictly think about what is needed to follow the instruction correctly on the website and if the trajectory reflects these steps.
2. Make sure your thought is enclosed in [thought] and [thought] tags.
3. Make sure your reward is enclosed in [reward] and [reward] tags and is a number from 1 to 5.
"""


def convert_to_uri(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_data_to_uri(image_array)


def get_trajectory_info(past_screenshots, past_actions):
    """
    Convert this into a dictionary
    """
    screenshot_uris = [
        {"type": "image_url", "image_url": {"url": convert_to_uri(screenshot)}}
        for screenshot in past_screenshots
    ]
    trajectory = []
    for idx, (screenshot, action) in enumerate(zip(screenshot_uris, past_actions)):
        trajectory.append({"type": "text", "text": "Screenshot {}:\n".format(idx + 1)})
        trajectory.append(screenshot)
        trajectory.append(
            {"type": "text", "text": "Action-{}: {}\n".format(idx + 1, action)}
        )
    return trajectory


def get_prompt(dirname):
    goal = pickle.load(gzip.open(f"{dirname}/goal_object.pkl.gz", "rb"))[0]["text"]
    num_screenshots = len(glob.glob(f"{dirname}/step_*.pkl.gz"))
    message_content = []
    if num_screenshots < 2:
        return None

    all_actions = []
    all_screenshots = []
    for i in range(num_screenshots - 1):
        step_curr = pickle.load(gzip.open(f"{dirname}/step_{i}.pkl.gz", "rb"))
        action_i = step_curr.action
        tstep_i = f"{dirname}/screenshot_step_{i}.png"
        all_actions.append(action_i)
        all_screenshots.append(tstep_i)

    trajectory_prompt = get_trajectory_info(all_screenshots, all_actions)
    input_prompt = [{"type": "text", "text": "# Objective: {}".format(goal)}]
    message_content = (
        input_prompt + [{"type": "text", "text": "# Trajectory:\n"}] + trajectory_prompt
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": message_content},
    ]


def parse_thought_and_reward(response):

    thought_count = response.count("[thought]")
    reward_count = response.count("[reward]")
    if thought_count == 2:
        thought_match = re.search(r"\[thought\](.*?)\[thought\]", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
    elif thought_count == 1:
        # just get everything after the first [thought] and before the first [reward]
        thought_match = re.search(r"\[thought\](.*?)\[reward\]", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
    else:
        thought = ""

    if reward_count == 2:
        reward_match = re.search(r"\[reward\](.*?)\[reward\]", response, re.DOTALL)
        reward = reward_match.group(1).strip() if reward_match else ""
    elif reward_count == 1:
        # just get everything after the first [reward] and before the end of the string
        reward_match = re.search(r"\[reward\](.*)", response, re.DOTALL)
        reward = reward_match.group(1).strip() if reward_match else ""
    else:
        reward = ""

    return {
        "thought": thought,
        "reward": reward,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a vision based reward model.")
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory containing the data files."
    )
    args = parser.parse_args()

    all_dirs = [d for d in glob.glob(os.path.join(args.dir, "*")) if os.path.isdir(d)]
    client = OpenAI(
        # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    for _dir in tqdm(all_dirs):
        # check if reward.pickle exists in the directory. if so, skip this directory
        if os.path.exists(os.path.join(_dir, "reward.pickle")):
            continue

        if not os.path.exists(os.path.join(_dir, "goal_object.pkl.gz")):
            continue

        prompt = get_prompt(_dir)
        if prompt is None:
            continue
        try:
            chunk = client.chat.completions.create(
                model="qwen-vl-max",
                messages=prompt,
            )
            response = chunk.choices[0].message.content

            response_dict = parse_thought_and_reward(response)

            response_dict["input_prompts"] = chunk.usage.prompt_tokens
            response_dict["output_tokens"] = chunk.usage.completion_tokens
            response_dict["raw_response"] = response
            with open(os.path.join(_dir, "reward.pickle"), "wb") as f:
                pickle.dump(response_dict, f)
        except Exception as e:
            print(f"Error processing directory {_dir}: {e}")
            continue

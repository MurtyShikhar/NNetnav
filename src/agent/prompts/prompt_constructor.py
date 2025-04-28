import json
import cv2
import re
from pathlib import Path
from typing import Any, TypedDict

from browser_env import Action, ActionParsingError, Trajectory
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput
import os
from agentlab.llm.llm_utils import ParseError
from agentlab.agents.visualwebarena.agent import image_data_to_uri
from agentlab.llm.chat_api import make_system_message, make_user_message

MOUSE_ICON_PATH = (
    "/u/scr/smurty/agents-with-exploration/public/assets/mouse-cursor-icon.png"
)


def extract_tags(text, keys):
    """Extract the content within tags for a list of keys.

    All text and keys will be converted to lowercase before matching.

    Returns:
        dict: A dictionary mapping each key to a list of subset in `text` that match the key.
    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        # make a pattern that matches the text between [key] and [key]
        pattern = rf"\[{key}\](.*?)\[{key}\]"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def parse_tags_raise(text, keys=(), optional_keys=(), merge_multiple=False):
    """A version of parse_tags that raises an exception if the parsing is not successful."""
    content_dict, valid, retry_message = parse_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def parse_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """Satisfy the parse api, extracts 1 match per key and validates that all keys are present

    Args:
        text (str): The input string containing the tags.
        keys (list[str]): The tags to extract the content from.
        optional_keys (list[str]): The tags to extract the content from, but are optional.
        merge_multiple (bool): Whether to merge multiple instances of the same key.

    Returns:
        dict: A dictionary mapping each key to a subset of `text` that match the key.
        bool: Whether the parsing was successful.
        str: A message to be displayed to the agent if the parsing was not successful.

    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if not key in content_dict:
            if not key in optional_keys:
                retry_messages.append(f"Missing the key <{key}> in the answer.")
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. You should have only one of them."
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message


def map_coordinates_to_dom(coordinates, elem_properties):
    """Map (x, y) coordinates to a DOM ID based on bounding boxes."""
    for dom_id in elem_properties:
        properties = elem_properties[dom_id]
        if "bbox" not in properties:
            continue
        if "set_of_marks" in properties and not properties["set_of_marks"]:
            continue
        if not properties["bbox"]:
            continue
        x, y, width, height = elem_properties[dom_id]["bbox"]
        if x <= coordinates[0] <= x + width and y <= coordinates[1] <= y + height:
            return dom_id
    return None


def extract_coordinates(action_text):
    """Extract (x, y) coordinates from an action string."""
    import re

    match = re.search(r"\((\d+),\s*(\d+)\)", action_text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def create_bgym_action_with_coordinates(action_str: str) -> tuple[str, dict]:
    """Convert from action_str to a bgym action space with coordinates.
    The bgym action space is defined as follows:
    noop(wait_ms: float = 1000)
    report_infeasible(reason: str)
    send_msg_to_user(text: str)
    """
    action_str = action_str.strip()
    action = (
        action_str.split("(")[0].strip()
        if "(" in action_str
        else action_str.split()[0].strip()
    )
    match action:
        case "click":
            match = re.match(
                r"click\s*\((\d+),\s*(\d+)\)\s*",
                action_str,
            )
            x, y = match.groups()
            coordinates = (int(x), int(y))
            return "mouse_click({}, {})".format(x, y), {
                "action_type": "click",
                "coordinates": coordinates,
            }
        case "dblclick":
            match = re.match(
                r"dblclick\s*\((\d+),\s*(\d+)\)\s*",
                action_str,
            )
            x, y = match.groups()
            coordinates = (int(x), int(y))
            return "mouse_dblclick({}, {})".format(x, y), {
                "action_type": "dblclick",
                "coordinates": coordinates,
            }
        case "type_no_clear":
            match = re.search(
                r"type_no_clear\s*\((\d+),\s*(\d+)\)\s*\[(.+)\]\s*",
                action_str,
            )
            x, y, text = match.groups()
            coordinates = (int(x), int(y))
            return "mouse_click({}, {})\nkeyboard_type('{}')".format(x, y, text), {
                "action_type": "type_no_clear",
                "text": text,
                "coordinates": coordinates,
            }
        case "hover":
            match = re.match(
                r"hover\s*\((\d+),\s*(\d+)\)\s*",
                action_str,
            )
            x, y = match.groups()
            coordinates = (int(x), int(y))
            return "mouse_move({}, {})".format(x, y), {
                "action_type": "hover",
                "coordinates": coordinates,
            }
        case "type":
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"
            # something like type (x, y) [text] [0] or type (x, y) [text] [1]
            match = re.search(
                r"type\s*\((\d+),\s*(\d+)\)\s*\[(.+)\]\s*\[(\d+)\]",
                action_str,
            )
            x, y, text, enter_flag = match.groups()
            coordinates = (int(x), int(y))
            # need to mouse_click(x, y) then keyboard_type(text) then keyboard_press(enter) if enter_flag == 1
            if enter_flag == "1":
                return (
                    "mouse_click({}, {})\nkeyboard_press('Meta+a')\nkeyboard_press('Backspace')\nkeyboard_type('{}')\nkeyboard_press('Enter')".format(
                        x, y, text
                    ),
                    {
                        "action_type": "type",
                        "coordinates": coordinates,
                        "text": text,
                        "enter_flag": enter_flag,
                    },
                )
            else:
                return (
                    "mouse_click({}, {})\nkeyboard_press('Meta+a')\nkeyboard_press('Backspace')\nkeyboard_type('{}')".format(
                        x, y, text
                    ),
                    {
                        "action_type": "type",
                        "coordinates": coordinates,
                        "text": text,
                        "enter_flag": enter_flag,
                    },
                )
        case "press":
            match = re.search(r"press ?\[(.+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid press action {action_str}")
            key_comb = match.group(1)
            return "keyboard_press('{}')".format(key_comb), {
                "action_type": "press",
                "key_comb": key_comb,
            }
        case "scroll":
            # up or down
            match = re.search(r"scroll ?\[?(up|down)\]?", action_str)
            if not match:
                raise ParseError(f"Invalid scroll action {action_str}")
            direction = match.group(1)
            if direction == "down":
                return "scroll(0, 100)", {
                    "action_type": "scroll",
                    "direction": direction,
                }
            else:
                return "scroll(0, -100)", {
                    "action_type": "scroll",
                    "direction": direction,
                }
        case "goto":
            match = re.search(r"goto ?\[(.+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid goto action {action_str}")
            url = match.group(1)
            return "goto('{}')".format(url), {
                "action_type": "goto",
                "url": url,
            }
        case "new_tab":
            return "new_tab()", {
                "action_type": "new_tab",
            }
        case "go_back":
            return "go_back()", {
                "action_type": "go_back",
            }
        case "go_forward":
            return "go_forward()", {
                "action_type": "go_forward",
            }
        case "tab_focus":
            match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid tab_focus action {action_str}")
            page_number = int(match.group(1))
            return "tab_focus({})".format(page_number), {
                "action_type": "tab_focus",
                "page_number": page_number,
            }
        case "close_tab":
            return "tab_close()", {
                "action_type": "close_tab",
            }
        case "stop":  # stop answer
            match = re.search(r"stop ?\[(.+)\]", action_str)
            if not match:  # some tasks don't require an answer
                answer = ""
            else:
                answer = match.group(1)
            return "send_msg_to_user('{}')".format(answer), {
                "action_type": "stop",
                "answer": answer,
            }

    raise ParseError(f"Invalid action {action_str}")


def create_bgym_action(action_str: str) -> str:
    """Convert from the webarena action space to the bgym action space. Lame, but needs to be done.
    The bgym action space is defined as follows:
    noop(wait_ms: float = 1000)
    report_infeasible(reason: str)
    send_msg_to_user(text: str)
    click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
    hover(bid: str)
    fill(bid: str, value: str)
    keyboard_press(key: str)
    scroll(delta_x: float, delta_y: float)
    tab_focus(index: int)
    new_tab()
    tab_close()
    go_back()
    go_forward()
    goto(url: str)
    """
    action_str = action_str.strip()
    action = (
        action_str.split("[")[0].strip()
        if "[" in action_str
        else action_str.split()[0].strip()
    )
    match action:
        case "click":
            match = re.search(r"click ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid click action {action_str}")
            element_id = match.group(1)
            return "click('{}')".format(element_id)
        case "hover":
            match = re.search(r"hover ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid hover action {action_str}")
            element_id = match.group(1)
            return "hover('{}')".format(element_id)
        case "type":
            # add default enter flag
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"

            match = re.search(r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid type action {action_str}")
            element_id, text, enter_flag = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if enter_flag == "1":
                return "fill('{}', '{}')\nkeyboard_press('Enter')".format(
                    element_id, text
                )
            else:
                return "fill('{}', '{}')".format(element_id, text)

            # deal with enter flag later
        case "press":
            match = re.search(r"press ?\[(.+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid press action {action_str}")
            key_comb = match.group(1)
            return "keyboard_press('{}')".format(key_comb)
        case "scroll":
            # up or down
            match = re.search(r"scroll ?\[?(up|down)\]?", action_str)
            if not match:
                raise ParseError(f"Invalid scroll action {action_str}")
            direction = match.group(1)
            return "scroll(0, 100)" if direction == "down" else "scroll(0, -100)"
        case "goto":
            match = re.search(r"goto ?\[(.+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid goto action {action_str}")
            url = match.group(1)
            return "goto('{}')".format(url)
        case "new_tab":
            return "new_tab()"
        case "go_back":
            return "go_back()"
        case "go_forward":
            return "go_forward()"
        case "tab_focus":
            match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
            if not match:
                raise ParseError(f"Invalid tab_focus action {action_str}")
            page_number = int(match.group(1))
            return "tab_focus({})".format(page_number)
        case "close_tab":
            return "tab_close()"
        case "stop":  # stop answer
            match = re.search(r"stop ?\[(.+)\]", action_str)
            if not match:  # some tasks don't require an answer
                answer = ""
            else:
                answer = match.group(1)
            return "send_msg_to_user('{}')".format(answer)

    raise ParseError(f"Invalid action {action_str}")


class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]


class PromptConstructor(object):
    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def __lt__(self, other):
        if not isinstance(other, PromptConstructor):
            return NotImplemented
        return self.instruction_path < other.instruction_path

    def get_lm_api_input(
        self, intro: str, examples: list[tuple[str, str]], current: str
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str

        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                if "mistral" in self.lm_config.model or "llama" in self.lm_config.model:
                    intro_message = intro
                    message = [{"role": "system", "content": intro_message}]
                else:
                    message = [{"role": "system", "content": intro}]
                    for x, y in examples:
                        message.append(
                            {
                                "role": "system",
                                "name": "example_user",
                                "content": x,
                            }
                        )
                        message.append(
                            {
                                "role": "system",
                                "name": "example_assistant",
                                "content": y,
                            }
                        )
                message.append({"role": "user", "content": current})
                return message
            elif self.lm_config.mode == "completion":
                message = f"{intro}\n\n"
                message += "Here are a few examples:\n"
                for example in examples:
                    message += f"Observation\n:{example[0]}\n\n"
                    message += f"Action: {example[1]}\n\n"
                message += "Now make prediction given the observation\n\n"
                message += f"Observation\n:{current}\n\n"
                message += "Action:"
                return message
            else:
                raise ValueError(
                    f"OpenAI models do not support mode {self.lm_config.mode}"
                )
        elif "huggingface" in self.lm_config.provider:
            # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L320
            if "Llama-2" in self.lm_config.model:
                if self.lm_config.mode == "chat":
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    BOS, EOS = "<s>", "</s>"
                    # adding the system message to be the starting of the first example
                    examples = [
                        (
                            B_SYS + intro + E_SYS + examples[0][0],
                            examples[0][1],
                        )
                    ] + examples[1:]
                    message = "".join(
                        [
                            f"{BOS}{B_INST} {x.strip()} {E_INST} {y.strip()} {EOS}"
                            for (x, y) in examples
                        ]
                    )
                    # add the current observation
                    message += f"{BOS}{B_INST} {current.strip()} {E_INST} {self.instruction['meta_data'].get('force_prefix', '')}"

                    return message
                else:
                    raise ValueError("Only chat mode is supported for Llama-2")
            else:
                raise ValueError(
                    f"Huggingface models do not support model_tag {self.lm_config.gen_config['model_tag']}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class DirectPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        if "page" in state_info["info"]:
            page = state_info["info"]["page"]
            url = page.url
        else:
            url = state_info["observation"]["url"]
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(f"Cannot parse action from response {response}")


class PassivePromptConstructor(PromptConstructor):
    """An LM that generates some output based on changes to the environment."""

    def __init__(self, instruction_path, lm_config, tokenizer):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(self, inputs, *args):
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]

        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            for key in inputs:
                inputs[key] = self.tokenizer.decode(self.tokenizer.encode(inputs[key])[:max_obs_length])  # type: ignore[arg-type]

        # replace each keyword in the template with inputs[keyword]
        inputs_for_keywords = {k: inputs[k] for k in keywords}

        current = template.format(**inputs_for_keywords)
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _parse_answer(self, response: str) -> dict:
        force_prefix = self.instruction["meta_data"].get("force_prefix", "")
        splitter = self.instruction["meta_data"].get("action_splitter", ":")
        response = f"{force_prefix}{response}"
        try:
            if splitter == ":":
                parsed_response = " ".join(response.split(":")[1:]).strip()
            elif splitter == "":
                parsed_response = response.strip()
            else:
                parsed_response = response.split(splitter)[1].strip()
            if self.answer_phrase:
                # remove the colon from the answer phrase
                answer_phrase = self.answer_phrase.replace(":", "")
                last_sent = parsed_response.split("\n")[-1]
                last_sent = re.sub(answer_phrase, "", last_sent, count=1).strip()
            else:
                last_sent = "n/a"
            return {
                "output": parsed_response,
                "raw_prediction": response,
                "answer": last_sent,
            }
        except Exception as e:
            raise ParseError(
                f"Cannot parse output from response {response}. Error: {e}"
            )


class StructuredPassivePromptConstructorBasic(PassivePromptConstructor):
    """
    The same as PassivePromptConstructor but the output is like this: [output] model_output [output]
    """

    def __init__(self, instruction_path, lm_config, tokenizer):
        super().__init__(instruction_path, lm_config, tokenizer)

    def _parse_answer(self, response):
        try:
            # first extract the stuff inside the think tag
            ans_dict = {}
            ans_dict["answer"] = parse_tags_raise(
                response, keys=["output"], merge_multiple=True
            )["output"]

            return {
                "raw_prediction": response,
                "answer": ans_dict["answer"],
            }
        except ParseError as e:
            raise ParseError(
                f"Error while parsing output\n: {e}\n"
                "Make sure your output is correctly formatted. In particular, make sure to wrap your output in the [output] tag."
            )


class StructuredPassivePromptConstructor(PassivePromptConstructor):
    """
    The same as PassivePromptConstructor but the output is like this: <think> reasoning step </think> <output> output </output>
    """

    def __init__(self, instruction_path, lm_config, tokenizer):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def _parse_answer(self, response):
        try:
            # first extract the stuff inside the think tag
            ans_dict = {}
            ans_dict["thought"] = parse_tags_raise(
                response, keys=["think"], merge_multiple=True
            )["think"]
            ans_dict["answer"] = parse_tags_raise(
                response, keys=["output"], merge_multiple=True
            )["output"]

            return {
                "output": "Thought: {}. Answer: {}".format(
                    ans_dict["thought"], ans_dict["answer"]
                ),
                "raw_prediction": response,
                "answer": ans_dict["answer"],
            }
        except ParseError as e:
            raise ParseError(
                f"Error while parsing output\n: {e}\n"
                "Make sure your output is correctly formatted."
            )


class CoTPromptConstructorBgym(PromptConstructor):
    """Same as a prompt constructor, but for bgym"""

    def __init__(self, instruction_path, lm_config, tokenizer):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]
        self.open_world = self.instruction["meta_data"].get("open_world", False)

    def construct(self, obs, meta_data) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        observation = obs["axtree_txt"]
        url = obs["url"]
        intent = obs["goal"]
        previous_action_str = meta_data["action_history"][-1]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            observation = self.tokenizer.decode(self.tokenizer.encode(observation)[:max_obs_length])  # type: ignore[arg-type]

        inputs_for_keywords = {
            "objective": intent,
            "url": url,
            "observation": observation,
            "previous_action": previous_action_str,
        }
        if "person_description" in meta_data:
            inputs_for_keywords["person_description"] = meta_data["person_description"]

        if "person_description" in meta_data:
            inputs_for_keywords["person_description"] = meta_data["person_description"]
        if "trajectory" in meta_data:
            inputs_for_keywords["trajectory"] = meta_data["trajectory"]

        if "past_actions" in keywords:
            all_actions = meta_data["action_history"]
            all_actions_with_step_counts = [
                "{}: {}".format(i + 1, a) for i, a in enumerate(all_actions)
            ]
            inputs_for_keywords["past_actions"] = "\n".join(
                all_actions_with_step_counts
            )

        inputs_for_keywords = {k: inputs_for_keywords[k] for k in keywords}
        current = template.format(**inputs_for_keywords)
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ParseError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        WA_REDDIT = os.getenv("WA_REDDIT", "")
        WA_SHOPPING = os.getenv("WA_SHOPPING", "")
        WA_SHOPPING_ADMIN = os.getenv("WA_SHOPPING_ADMIN", "")
        WA_GITLAB = os.getenv("WA_GITLAB", "")
        WA_WIKIPEDIA = os.getenv("WA_WIKIPEDIA", "")
        WA_MAP = os.getenv("WA_MAP", "")
        WA_HOMEPAGE = os.getenv("WA_HOMEPAGE", "")

        URL_MAPPINGS_BGYM = {
            WA_REDDIT: "http://reddit.com",
            WA_SHOPPING: "http://onestopmarket.com",
            WA_SHOPPING_ADMIN: "http://luma.com/admin",
            WA_GITLAB: "http://gitlab.com",
            WA_WIKIPEDIA: "http://wikipedia.org",
            WA_MAP: "http://openstreetmap.org",
            WA_HOMEPAGE: "http://homepage.com",
        }

        for i, j in URL_MAPPINGS_BGYM.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response

    def _parse_answer(self, response: str) -> dict:
        try:
            parsed_response = self.extract_action(response)
            # for the bgym action we need to map goto links to the local links
            if self.open_world:
                # do not remap urls
                bgym_action = create_bgym_action(parsed_response)
            else:
                bgym_action = create_bgym_action(self.map_url_to_local(parsed_response))
            return {
                "action": parsed_response,
                "bgym_action": bgym_action,
                "raw_prediction": response,
            }
        except Exception as e:
            raise ParseError(
                f"Cannot parse action from response {response}. Error: {e}"
            )


class CoTPromptConstructVision(CoTPromptConstructorBgym):
    """Same as a bgym prompt constructor, but for vision"""

    def get_trajectory_info(self, past_screenshots, past_actions):
        """
        Convert this into a dictionary
        """
        screenshot_uris = [
            {"type": "image_url", "image_url": {"url": image_data_to_uri(screenshot)}}
            for screenshot in past_screenshots
        ]

        trajectory = []
        for idx, (screenshot, action) in enumerate(zip(screenshot_uris, past_actions)):
            trajectory.append(
                {"type": "text", "text": "Screenshot {}:\n".format(idx + 1)}
            )
            trajectory.append(screenshot)
            trajectory.append(
                {"type": "text", "text": "Action-{}: {}\n".format(idx + 1, action)}
            )
        return trajectory

    def construct(self, obs, meta_data) -> APIInput:
        self.bbox_info = meta_data["curr_obs_bboxes"]
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        if self.instruction["meta_data"].get("use_som", True):
            screenshot = obs["screenshot_som"]
        else:
            # fallback to the original screenshot if som is not used
            screenshot = obs["screenshot"]

        self.screenshot = screenshot
        url = obs["url"]
        intent = obs["goal"]

        inputs_for_keywords = {
            "objective": intent,
            "url": url,
            "observation": "<see screenshot below>",
        }

        past_screenshots = meta_data["past_screenshots"]
        if len(past_screenshots) == 0:
            trajectory_info = [{"type": "text", "text": "None\n"}]
        else:
            past_actions = meta_data["action_history"][
                1:
            ]  # get rid of the "None" padding
            trajectory_info = self.get_trajectory_info(past_screenshots, past_actions)

        inputs_for_keywords = {k: inputs_for_keywords[k] for k in keywords}
        current = template.format(**inputs_for_keywords)
        assert all([f"{{k}}" not in current for k in keywords])

        intro_message = []
        intro_message.append({"type": "text", "text": intro})
        assert not examples  # make sure there are no examples
        user_messages = []
        user_messages.append({"type": "text", "text": current})
        user_messages.append({"type": "text", "text": "Trajectory-so-far:\n"})
        user_messages.extend(trajectory_info)

        user_messages.extend(
            [
                {"type": "text", "text": "Current Observation:\n"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_to_uri(screenshot)},
                },
            ]
        )

        prompt = [
            make_system_message(intro_message),
            make_user_message(user_messages),
        ]

        return prompt

    def mark_coordinates(self, screenshot, coordinates):
        # Load the mouse icon with alpha channel (transparency)
        mouse_icon = cv2.imread(
            MOUSE_ICON_PATH, cv2.IMREAD_UNCHANGED
        )  # Shape: (h, w, 4)
        if mouse_icon is None:
            print("Mouse icon not found!")
            return
        marked = screenshot.copy()
        x, y = coordinates
        # Resize mouse icon if needed
        icon_h, icon_w = mouse_icon.shape[:2]
        if icon_h > 30 or icon_w > 30:
            mouse_icon = cv2.resize(mouse_icon, (30, 30), interpolation=cv2.INTER_AREA)
            icon_h, icon_w = mouse_icon.shape[:2]

        # Calculate top-left corner of where to place the icon
        top_left_x = x - icon_w // 2
        top_left_y = y - icon_h // 2

        # Make sure the coordinates are within bounds
        top_left_x = max(0, min(top_left_x, marked.shape[1] - icon_w))
        top_left_y = max(0, min(top_left_y, marked.shape[0] - icon_h))

        # Split the icon into color and alpha channels
        icon_rgb = mouse_icon[:, :, :3]
        icon_alpha = mouse_icon[:, :, 3] / 255.0

        # Region of interest on the base image
        roi = marked[top_left_y : top_left_y + icon_h, top_left_x : top_left_x + icon_w]

        # Blend the icon with the ROI using the alpha channel
        for c in range(3):  # For each color channel
            roi[:, :, c] = (
                roi[:, :, c] * (1 - icon_alpha) + icon_rgb[:, :, c] * icon_alpha
            )

        marked[top_left_y : top_left_y + icon_h, top_left_x : top_left_x + icon_w] = roi
        return marked

    def _parse_answer_coordinate(self, response: str) -> dict:
        """
        Answer parser if the low-level controller directly allows for executing actions based on coordinates.
        """
        try:
            parsed_response = self.extract_action(response)
            bgym_action, orig_action = create_bgym_action_with_coordinates(
                parsed_response
            )
            return {
                "action": parsed_response,
                "bgym_action": bgym_action,
                "raw_prediction": response,
                "orig_action": orig_action,
            }
        except Exception as e:
            raise ParseError(
                f"\nCannot parse action from response {response}. Error: {e}"
            )

    def _parse_answer_bid(self, response: str) -> dict:
        try:
            parsed_response = self.extract_action(response)
            # for the bgym action we need to map goto links to the local links
            use_som = self.instruction["meta_data"].get("use_som", True)
            if not use_som:
                # we will handle later
                action_type = parsed_response.split(" ")[0]
                if action_type in ["click", "type", "hover"]:
                    # get dom id
                    match = re.match(
                        r"(click|type|hover)\s*\((\d+),\s*(\d+)\)\s*(?:\[([^\]]+)\])?\s*(?:\[(\d+)\])?",
                        parsed_response,
                    )
                    if not match:
                        raise ParseError(
                            f"\nCannot parse the coordinates from {parsed_response}. Please check the format."
                        )
                    action_type, x, y, text, index = match.groups()
                    coordinates = (int(x), int(y))
                    index = int(index) if index else None

                    dom_id = map_coordinates_to_dom(coordinates, self.bbox_info)
                    if action_type in ["click", "hover"]:
                        for_bgym = f"{action_type} [{dom_id}]"
                    else:
                        for_bgym = f"type [{dom_id}] [{text}]"
                        if index is not None:
                            # append the index to the type action
                            for_bgym += f" [{index}]"
                    if dom_id is None:
                        raise ParseError(
                            f"\nThere is no interactable element at coordinates ({x}, {y}) in the screenshot. Please check the screenshot and only issue actions for valid elements."
                        )
                    # remap parsed_response to the for_bgym action
                    parsed_response = for_bgym
                    orig_action = {
                        "action_type": action_type,
                        "dom_id": dom_id,
                        "text": text,
                        "index": index,
                        "coordinates": coordinates,
                    }
                else:
                    # coordinates not applicable
                    orig_action = {"action_type": action_type}
                bgym_action = create_bgym_action(parsed_response)

            elif self.open_world:
                # do not remap urls
                bgym_action = create_bgym_action(parsed_response)
            else:
                bgym_action = create_bgym_action(self.map_url_to_local(parsed_response))

            if not use_som:
                return {
                    "action": parsed_response,
                    "bgym_action": bgym_action,
                    "raw_prediction": response,
                    "orig_action": orig_action,
                }
            else:
                return {
                    "action": parsed_response,
                    "bgym_action": bgym_action,
                    "raw_prediction": response,
                }
        except Exception as e:
            raise ParseError(
                f"Cannot parse action from response {response}. Error: {e}"
            )


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        if "page" in state_info["info"]:
            page = state_info["info"]["page"]
            url = page.url
        else:
            url = state_info["observation"]["url"]

        previous_action_str = meta_data["action_history"][-1]

        inputs_for_keywords = {
            "objective": intent,
            "url": url,
            "observation": obs,
            "previous_action": previous_action_str,
        }
        if "person_description" in meta_data:
            inputs_for_keywords["person_description"] = meta_data["person_description"]
        if "history" in trajectory[-1]:
            inputs_for_keywords["trajectory"] = trajectory[-1]["history"]

        if "past_actions" in keywords:
            all_actions = meta_data["action_history"]
            all_actions_with_step_counts = [
                "{}: {}".format(i + 1, a) for i, a in enumerate(all_actions)
            ]
            inputs_for_keywords["past_actions"] = "\n".join(
                all_actions_with_step_counts
            )

        inputs_for_keywords = {k: inputs_for_keywords[k] for k in keywords}
        current = template.format(**inputs_for_keywords)
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )

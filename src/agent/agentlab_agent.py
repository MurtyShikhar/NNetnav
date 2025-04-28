"""
Agent that is compatible with the BrowserGym / Agentlab framework.
"""

from dataclasses import dataclass, asdict
from browsergym.experiments.agent import Agent as BrowserGymAgent
from browsergym.experiments.agent import AgentInfo
import re
import os

from agentlab.llm.chat_api import BaseModelArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.agent_configs import (
    CHAT_MODEL_ARGS_DICT,
    FLAGS_GPT_4o,
)
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer
import argparse
import logging

from agent.prompts import *
from agentlab.agents.agent_args import AgentArgs
import json
import bgym
from copy import deepcopy
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from browser_env.helper_functions import (
    get_action_description_bgym,
    get_action_description_with_coordinates,
)
from browser_env.actions import (
    create_id_based_action,
    create_playwright_action,
)

from browsergym.core.registration import register_task
from nnetnav_registry import WebArenaOpenEnded, NNetNavOpenEndedTask
import nnetnav_registry
import webvoyager_registry


logger = logging.getLogger(__name__)

generic_flags_webarena = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=True,
        action_set="webarena",
        long_description=False,
        individual_examples=False,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=None,
    be_cautious=True,
    extra_instructions=None,
)

FLAGS_VLM = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=True,
        use_som=True,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=bgym.HighLevelActionSetArgs(
            subsets=["webarena"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=False,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=None,
    be_cautious=True,
    extra_instructions=None,
)


def convert_to_description(changelogs):
    """
    returns a natural language description of all the changes to states
    """
    i = 0
    descriptions = []
    for log in changelogs:
        cstr = "Step: " + str(i) + "\n"
        cstr += log
        descriptions.append(cstr)
        i += 1
    return "\n\n".join(descriptions)


def get_prompt_constructor(args, prompt_path):
    llm_config = lm_config.construct_llm_config(args)
    with open(prompt_path) as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        prompt_path, lm_config=llm_config, tokenizer=tokenizer
    )

    return prompt_constructor


@dataclass
class ExplorationAgentFactory(GenericAgentArgs):
    args: argparse.Namespace = None
    task_args: tuple = None
    persona_str: str = None
    exploration_prompt_constructor_path: str = None
    change_summarizer_prompt_constructor_path: str = None
    trajectory_labeler_prompt_constructor_path: str = None
    outcome_reward_model_prompt_constructor_path: str = None

    def prepare(self):
        return

    def close(self):
        return

    def make_agent(self):
        # first register the task
        gym_id, task_kwargs = self.task_args
        if "webarena_nnetnav" in gym_id:
            register_task(gym_id, WebArenaOpenEnded, task_kwargs=task_kwargs)
        else:
            register_task(gym_id, NNetNavOpenEndedTask, task_kwargs=task_kwargs)

        args = self.args
        llm_config = lm_config.construct_llm_config(args)
        benchmark = bgym.DEFAULT_BENCHMARKS["webarena"]()
        self.set_benchmark(benchmark, False)
        self.flags.action.action_set.multiaction = True

        exploration_prompt_constructor = get_prompt_constructor(
            args, self.exploration_prompt_constructor_path
        )
        change_summarizer_prompt_constructor = get_prompt_constructor(
            args, self.change_summarizer_prompt_constructor_path
        )
        trajectory_labeler_prompt_constructor = get_prompt_constructor(
            args, self.trajectory_labeler_prompt_constructor_path
        )
        outcome_reward_model_prompt_constructor = get_prompt_constructor(
            args, self.outcome_reward_model_prompt_constructor_path
        )

        agent = NNetNavExplorerAgent(
            action_set_tag=args.action_set_tag,
            prune_at=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
            persona_str=self.persona_str,
            lm_config=llm_config,
            prompts={
                "exploration": exploration_prompt_constructor,
                "change_summarizer": change_summarizer_prompt_constructor,
                "trajectory_labeler": trajectory_labeler_prompt_constructor,
                "outcome_reward_model": outcome_reward_model_prompt_constructor,
            },
            flags=self.flags,
            chat_llm=self.chat_model_args.make_model(),
        )
        return agent


@dataclass
class AgentFactory(GenericAgentArgs):
    args: argparse.Namespace = None

    def prepare(self):
        return

    def close(self):
        return

    def make_agent(self):
        args = self.args
        llm_config = lm_config.construct_llm_config(args)
        assert args.agent_type == "prompt"

        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        if prompt_constructor.instruction["meta_data"].get("use_som", True):
            benchmark = bgym.DEFAULT_BENCHMARKS["webarena"]()
        else:
            benchmark = bgym.DEFAULT_BENCHMARKS["coordinate_webarena"]()
        self.set_benchmark(benchmark, False)
        self.flags.action.action_set.multiaction = True
        agent = NNetNavBrowserGymAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            flags=self.flags,
            chat_llm=self.chat_model_args.make_model(),
        )
        return agent


@dataclass
class VLMAgentFactory(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    max_retry: int = 4
    args: argparse.Namespace = None

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"VLMAgent-{self.chat_model_args.model_name}".replace(
                "/", "_"
            )
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        # for backward compatibility with old traces
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        # verify if we can remove this
        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self) -> bgym.Agent:
        args = self.args
        llm_config = lm_config.construct_llm_config(args)

        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        if prompt_constructor.instruction["meta_data"].get("use_som", True):
            benchmark = bgym.DEFAULT_BENCHMARKS["webarena"]()
        else:
            benchmark = bgym.DEFAULT_BENCHMARKS["coordinate_webarena"]()
        self.set_benchmark(benchmark, False)
        self.flags.action.action_set.multiaction = True
        agent = NNetNavBrowserGymAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            flags=self.flags,
            chat_llm=self.chat_model_args.make_model(),
        )
        return agent


class LMModule:
    """
    A generic module class to instantiate various LM modules needed in NNetNav
    """

    def __init__(
        self, chat_llm, flags, prompt_constructor, max_retry=4, fail_message=""
    ):
        self.chat_llm = chat_llm
        self.flags = flags
        self.max_retry = max_retry
        self.prompt_constructor = prompt_constructor
        self.fail_message = fail_message

    def __call__(self, obs: dict) -> dict:
        prompt = self.prompt_constructor.construct(obs)
        try:
            chat_messages = Discussion(prompt)
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=self.prompt_constructor._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                answer=self.fail_message,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )
        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]
        return ans_dict


class NNetNavExplorerAgent(BrowserGymAgent):
    """
    Agent that uses the NNetNav algorithm to explore the web.
    """

    def __init__(
        self, action_set_tag, prune_at, persona_str, lm_config, prompts, flags, chat_llm
    ):
        """
        action_set_tag: str
            The action set tag to use (mostly for legacy reasons)
        prune_at: list
            At what depths to prune exploration
        persona_str: str
            The persona string for the exploration policy
        lm_config: dict
            The language model configuration (also for legacy reasons, should be removed)
        prompts: dict
            The prompts for the exploration policy, change summarizer, trajectory labeler, and outcome reward model.
            has keys "exploration", "change_summarizer", "trajectory_labeler", and "outcome_reward_model"
        flags:
            Agentlab / browsergym object that controls the action space, observation space, etc.
        chat_llm:
            The base LLM backbone for all components.
        """
        self.flags = flags
        self.prune_at = prune_at
        self.action_set = self.flags.action.action_set.make_action_set()
        self.chat_llm = chat_llm
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)
        self.action_set_tag = action_set_tag
        self.lm_config = lm_config
        # get rid of max_obs_length in self.lm_config because bgym does not use it
        if "max_obs_length" in self.lm_config.gen_config:
            del self.lm_config.gen_config["max_obs_length"]
        self.max_retry = lm_config.gen_config["max_retry"]
        self.prompts = prompts
        self.modules = {
            key: LMModule(chat_llm, flags, prompts[key], max_retry=self.max_retry)
            for key in prompts
            if key != "exploration"
        }
        self.persona_str = persona_str

        self.reset(seed=None)

    def get_action_description_helper(self, obs, action):
        # get all bids
        all_bids = [o for o in obs["axtree_object"]["nodes"] if "browsergym_id" in o]
        bid_dict = {}
        for o in all_bids:
            if "name" in o:
                bid_dict[o["browsergym_id"]] = o["name"]["value"]
            else:
                bid_dict[o["browsergym_id"]] = ""
        action_splitter = self.prompts["exploration"].instruction["meta_data"][
            "action_splitter"
        ]
        action_str = get_action_description_bgym(
            action,
            bid_dict,
            action_set_tag=self.action_set_tag,
            action_splitter=action_splitter,
        )

        return action_str

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def reset(self, seed=None):
        self.seed = seed
        self.thoughts = []
        self.actions = ["None"]
        self.obs_history = []
        self.past_transitions = []
        self.past_subtasks = ["None"]

    def early_stop(self):
        """
        if last 3 actions are the same, then stop unless its a scroll action
        """
        if len(self.actions) < 3:
            return False
        if self.actions[-1] == self.actions[-2] == self.actions[-3]:
            if "scroll" in self.actions[-1]:
                return False
            return True

    def get_action(self, obs: dict) -> dict:
        self.obs_history.append(obs)

        if len(self.obs_history) > 1:
            change_summary = self.modules["change_summarizer"](
                {
                    "init_observation": self.obs_history[-2]["axtree_txt"],
                    "final_observation": self.obs_history[-1]["axtree_txt"],
                    "action": self.actions[-1],
                }
            )
            self.past_transitions.append(change_summary["output"])

        # first check if the current time step is a pruning time step
        curr_time_step = len(self.obs_history) - 1
        trajectory_description = convert_to_description(self.past_transitions)
        if curr_time_step in self.prune_at:
            # now check if the inferred sub-task is meaningful
            trajectory_label = self.modules["trajectory_labeler"](
                {
                    "trajectory": trajectory_description,
                }
            )
            # add logging
            logger.info(f"[Sub-task label]: {trajectory_label['answer']}")
            reward = self.modules["outcome_reward_model"](
                {
                    "trajectory": trajectory_description,
                    "instruction": trajectory_label["answer"],
                    "previous_subtask": self.past_subtasks[-1],
                },
            )
            # add logging
            logger.info(f"[Reward]: {reward['answer']}")
            logger.info(f"[Reward Reasoning]: {reward['output']}")
            self.past_subtasks.append(trajectory_label["answer"])
            if int(reward["answer"]) < 4:
                # if the reward is zero, then we should not explore further
                ans_dict = dict(
                    action="stop[exit]",
                    bgym_action="send_msg_to_user('exit')",
                )
                agent_info = AgentInfo(
                    think="Pruning exploration",
                    chat_messages=[],
                    extra_info={
                        "webarena_action_ds": None,
                        "trajectory_label": trajectory_label,
                        "reward": reward,
                        "summary": self.past_transitions,
                    },
                )
                return ans_dict["bgym_action"], agent_info
        else:
            trajectory_label = None
            reward = None

        exploration_prompt_constructor = self.prompts["exploration"]
        exploration_agent_prompt = self.prompts["exploration"].construct(
            obs,
            meta_data={
                "action_history": self.actions,
                "trajectory": trajectory_description,
                "person_description": self.persona_str,
            },
        )
        # TODO: we want to return none if it turns out that this action is not leading to a meaningful sub-task
        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long
            chat_messages = Discussion(exploration_agent_prompt)
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=exploration_prompt_constructor._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action="stop['exit due to parse error']",
                bgym_action="send_msg_to_user('exit due to action parse error')",
                raw_prediction="Cannot correctly parse the action",
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )
        if self.action_set_tag == "id_accessibility_tree":
            webarena_action_ds = create_id_based_action(ans_dict["action"])
        elif self.action_set_tag == "playwright":
            webarena_action_ds = create_playwright_action(ans_dict["action"])
        else:
            raise ValueError(f"Unknown action type {self.action_set_tag}")
        ans_dict["webarena_action_ds"] = webarena_action_ds
        self.actions.append(
            self.get_action_description_helper(obs, ans_dict["webarena_action_ds"])
        )

        if self.early_stop():
            ans_dict["bgym_action"] = "send_msg_to_user('exit')"
            agent_info = AgentInfo(
                think="Early stopping",
                chat_messages=exploration_agent_prompt,
                extra_info={
                    "webarena_action_ds": webarena_action_ds,
                    "trajectory_label": trajectory_label,
                    "reward": reward,
                    "summary": self.past_transitions,
                },
            )
            return ans_dict["bgym_action"], agent_info

        agent_info = AgentInfo(
            think=ans_dict["raw_prediction"],
            chat_messages=exploration_agent_prompt,
            extra_info={
                "webarena_action_ds": webarena_action_ds,
                "trajectory_label": trajectory_label,
                "reward": reward,
                "summary": self.past_transitions,
            },
        )
        return ans_dict["bgym_action"], agent_info


def extract_text(action_text):
    """Extract text input from a type action string."""
    import re

    match = re.search(r"\[([^\]]+)\]", action_text)
    return match.group(1) if match else ""


class NNetNavBrowserGymAgent(BrowserGymAgent):
    def __init__(self, action_set_tag, lm_config, prompt_constructor, flags, chat_llm):
        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self.chat_llm = chat_llm

        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)
        self.action_set_tag = action_set_tag
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.max_retry = lm_config.gen_config["max_retry"]
        homepage = os.getenv("WA_HOMEPAGE", "https://www.google.com")
        logger.info(f"Homepage: {homepage}")
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def reset(self, seed=None):
        self.seed = seed
        self.thoughts = []
        self.actions = ["None"]
        self.obs_history = []
        self.past_transitions = []
        self.curr_mouse_coords = (0, 0)

    def get_action(self, obs: dict) -> dict:
        # if we are not using SoM, then we are using a coordinate-based action space
        use_continuous_action_space = not self.prompt_constructor.instruction[
            "meta_data"
        ].get("use_som", True)
        screenshot_key = (
            "screenshot" if use_continuous_action_space else "screenshot_som"
        )
        past_observations = [
            obs[screenshot_key] for obs in self.obs_history
        ]  # all but the current observation

        if use_continuous_action_space:
            # draw mouse cursor on the screenshot
            screenshot = obs[screenshot_key]
            # draw the mouse cursor on the screenshot
            screenshot = self.prompt_constructor.mark_coordinates(
                screenshot, self.curr_mouse_coords
            )
            obs[screenshot_key] = screenshot
        else:
            screenshot = obs[screenshot_key]

        prompt = self.prompt_constructor.construct(
            obs,
            meta_data={
                "curr_obs_bboxes": obs.get(
                    "extra_element_properties", {}
                ),  # for mapping coordinates
                "action_history": self.actions,
                "past_screenshots": past_observations,  # pass the past screenshots for context
            },
        )
        lm_config = self.lm_config

        if use_continuous_action_space:
            answer_parser = self.prompt_constructor._parse_answer_coordinate
        else:
            answer_parser = self.prompt_constructor._parse_answer

        try:
            chat_messages = Discussion(prompt)
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=answer_parser,
            )
            ans_dict["busted_retry"] = 0
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action="stop[parsing error]",
                bgym_action="send_msg_to_user('parsing error')",
                raw_prediction="Cannot correctly parse the action",
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        # create a webarena_action_ds IF
        if not use_continuous_action_space:
            if self.action_set_tag == "id_accessibility_tree":
                webarena_action_ds = create_id_based_action(ans_dict["action"])
            elif self.action_set_tag == "playwright":
                webarena_action_ds = create_playwright_action(ans_dict["action"])
            else:
                raise ValueError(f"Unknown action type {self.action_set_tag}")
            ans_dict["webarena_action_ds"] = webarena_action_ds
            if "orig_action" in ans_dict and "coordinates" in ans_dict["orig_action"]:
                coordinates = ans_dict["orig_action"]["coordinates"]
                self.curr_mouse_coords = coordinates
            else:
                coordinates = None
            self.actions.append(
                self.get_action_description_helper(
                    obs, ans_dict["webarena_action_ds"], coordinates
                )
            )
        else:
            self.actions.append(ans_dict["action"])
            if "orig_action" in ans_dict and "coordinates" in ans_dict["orig_action"]:
                coordinates = ans_dict["orig_action"]["coordinates"]
                self.curr_mouse_coords = coordinates

            webarena_action_ds = None
        self.obs_history.append(obs)
        agent_info = AgentInfo(
            think=ans_dict["raw_prediction"],
            chat_messages=prompt,
            extra_info={
                "webarena_action_ds": webarena_action_ds,
                "curr_mouse_coords": self.curr_mouse_coords,
                "interaction": screenshot,
            },
        )
        return ans_dict["bgym_action"], agent_info

    def get_action_description_helper(self, obs, action, coordinates):
        # get all bids
        all_bids = [o for o in obs["axtree_object"]["nodes"] if "browsergym_id" in o]
        bid_dict = {}
        for o in all_bids:
            if "name" in o:
                bid_dict[o["browsergym_id"]] = o["name"]["value"]
            else:
                bid_dict[o["browsergym_id"]] = ""
        action_splitter = self.prompt_constructor.instruction["meta_data"][
            "action_splitter"
        ]
        if coordinates is not None:
            action_str = get_action_description_with_coordinates(
                action,
                bid_dict,
                action_set_tag=self.action_set_tag,
                action_splitter=action_splitter,
                coordinates=coordinates,
            )
        else:
            action_str = get_action_description_bgym(
                action,
                bid_dict,
                action_set_tag=self.action_set_tag,
                action_splitter=action_splitter,
            )

        return action_str

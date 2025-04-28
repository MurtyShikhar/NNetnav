import playwright.sync_api
import json

# we use a global playwright instance
_PLAYWRIGHT = None


def _set_global_playwright(pw: playwright.sync_api.Playwright):
    global _PLAYWRIGHT
    _PLAYWRIGHT = pw


def _get_global_playwright():
    global _PLAYWRIGHT
    if not _PLAYWRIGHT:
        pw = playwright.sync_api.sync_playwright().start()
        _set_global_playwright(pw)

    return _PLAYWRIGHT


# register the open-ended task
from browsergym.core.registration import register_task
from .webarena_openended_task import WebArenaOpenEnded, NNetNavOpenEndedTask


# get the current path, and use that to load configs located in the same directory
def get_configs():
    import os

    current_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(current_path, "NNetNav6k_configs/configs.json")
    return json.load(open(config_path))


def get_openweb_configs():
    import os

    current_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(current_path, "NNetNav_OW_Dec24_configs/configs.json")
    return json.load(open(config_path))


configs = get_configs()

ALL_OPENENDED_WEBARENA_TASK_IDS = []
ALL_OPENENDED_WEBARENA_TASK_IDS_MAPS = []
ALL_OPENENDED_WEBARENA_TASK_IDS_NO_MAPS = []
ALL_OPENWEB_TASK_IDS = []

for idx, _c in enumerate(configs):
    gym_id = f"webarena_openended_{idx}"
    register_task(
        gym_id,
        WebArenaOpenEnded,
        task_kwargs={"config_str": json.dumps(_c), "task_id": idx},
    )
    ALL_OPENENDED_WEBARENA_TASK_IDS.append(gym_id)
    if "map" in _c["sites"]:
        ALL_OPENENDED_WEBARENA_TASK_IDS_MAPS.append(gym_id)
    else:
        ALL_OPENENDED_WEBARENA_TASK_IDS_NO_MAPS.append(gym_id)

openweb_configs = get_openweb_configs()
for idx, _c in enumerate(openweb_configs):
    gym_id = f"openweb_{idx}"
    register_task(
        gym_id,
        NNetNavOpenEndedTask,
        task_kwargs={"start_url": _c["start_url"], "goal": _c["goal"]},
    )
    ALL_OPENWEB_TASK_IDS.append(gym_id)

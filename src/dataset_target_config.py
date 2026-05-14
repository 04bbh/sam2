"""Dataset-specific target descriptions and temporal range categories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetTargetConfig:
    name: str
    target_desc: str
    full_video_range_categories: frozenset[str]
    action_range_categories: frozenset[str]


SHT_ORIGINAL_TARGET_DESC = (
    "打斗，骑自行车，骑摩托车，机动车，跳远，抢夺，小推车，翻越栏杆，摔倒，"
    "向上抛掷物品，滑滑板，快速奔跑，挥舞长棍"
)

SHT_NEW_TARGET_DESC = (
    "打斗，骑自行车，骑摩托车，机动车，跳远，抢夺，小推车，摔倒，"
    "向上抛掷物品，滑滑板（出现滑板车），快速奔跑（双脚离地），挥舞长棍"
)

UBN_TARGET_DESC = (
    "快速奔跑，在人行道驾驶小汽车，躺下，车祸，着火，烟雾，打斗"
)
# "奔跑，在人行道驾驶机动车，躺下，走路摇摇晃晃，行人与车辆相撞，车祸，跳跃，着火，烟雾，乱穿马路，打斗"

UCF_TARGET_DESC = (
    "逮捕，着火，火光，纵火，打斗，车祸，爆炸，烟雾，抢劫，枪击，蓄意破坏，偷窃，夜晚在柜台边行窃"
)
XD_TARGET_DESC = (
    "枪击，爆炸，烟雾，打斗，暴乱，虐待，车辆撞击"
)


DEFAULT_FULL_VIDEO_RANGE_CATEGORIES = frozenset(
    {
        "骑自行车",
        "骑摩托车",
        "机动车",
        "小推车",
        "推车",
        "垃圾推车",
    }
)
DEFAULT_ACTION_RANGE_CATEGORIES = frozenset(
    {
        "打斗",
        "跳跃",
        "抢夺",
        "翻越栏杆",
        "摔倒",
        "奔跑",
        "快速奔跑",
        "追逐",
        "挥舞物品",
        "滑滑板",
        "滑滑板的人",
        "向上抛掷物品",
        "捡起掉落的物品",

    }
)


UBN_FULL_VIDEO_RANGE_CATEGORIES = frozenset(
    {
        "烟雾",
        "着火",
    }
)
UBN_ACTION_RANGE_CATEGORIES = frozenset(
    {
        "快速奔跑",
        "在人行道驾驶机动车",
        "躺下",
        "车祸",
        "打斗",
    }
)

UCF_FULL_VIDEO_RANGE_CATEGORIES = frozenset(
    {
        
    }
)
UCF_ACTION_RANGE_CATEGORIES = frozenset(
    {
        #虐待动物，逮捕，着火，单方面袭击，打斗，车辆撞击，爆炸，烟雾，抢劫，枪击，蓄意破坏，偷窃
        "虐待动物",
        "逮捕",
        "着火",
        "单方面袭击",
        "打斗",
        "车辆撞击",
        "爆炸",
        "烟雾",
        "抢劫",
        "枪击",
        "蓄意破坏",
        "偷窃",
    }
)

XD_FULL_VIDEO_RANGE_CATEGORIES = frozenset(
    {
        
    }
)
XD_ACTION_RANGE_CATEGORIES = frozenset(
    {
        # 枪击，爆炸，烟雾，打斗，暴乱，虐待，车辆撞击
        "枪击",
        "爆炸",
        "烟雾",
        "打斗",
        "暴乱",
        "虐待",
        "车辆撞击",

    }
)


DATASET_TARGET_CONFIGS: dict[str, DatasetTargetConfig] = {
    "sht_original": DatasetTargetConfig(
        name="sht_original",
        target_desc=SHT_ORIGINAL_TARGET_DESC,
        full_video_range_categories=DEFAULT_FULL_VIDEO_RANGE_CATEGORIES,
        action_range_categories=DEFAULT_ACTION_RANGE_CATEGORIES,
    ),
    "sht_new": DatasetTargetConfig(
        name="sht_new",
        target_desc=SHT_NEW_TARGET_DESC,
        full_video_range_categories=DEFAULT_FULL_VIDEO_RANGE_CATEGORIES,
        action_range_categories=DEFAULT_ACTION_RANGE_CATEGORIES,
    ),
    "ubnormal": DatasetTargetConfig(
        name="ubnormal",
        target_desc=UBN_TARGET_DESC,
        full_video_range_categories=UBN_FULL_VIDEO_RANGE_CATEGORIES,
        action_range_categories=UBN_ACTION_RANGE_CATEGORIES,
    ),
    "xd": DatasetTargetConfig(
        name="xd",
        target_desc=XD_TARGET_DESC,
        full_video_range_categories=XD_FULL_VIDEO_RANGE_CATEGORIES,
        action_range_categories=XD_ACTION_RANGE_CATEGORIES,
    ),
    "ucf": DatasetTargetConfig(
        name="ucf",
        target_desc=UCF_TARGET_DESC,
        full_video_range_categories=UCF_FULL_VIDEO_RANGE_CATEGORIES,
        action_range_categories=UCF_ACTION_RANGE_CATEGORIES,
    ),
}

DATASET_NAME_ALIASES = {
    "sht": "sht_original",
    "shanghaitech": "sht_original",
    "shanghai_tech": "sht_original",
    "sht_original": "sht_original",
    "sht_new": "sht_new",
    "ubnormal": "ubnormal",
    "xd": "xd",
    "ucf": "ucf",
}


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("data.dataset_name must be a non-empty string")
    return DATASET_NAME_ALIASES.get(normalized, normalized)


def get_dataset_target_config(dataset_name: str) -> DatasetTargetConfig:
    normalized = normalize_dataset_name(dataset_name)
    config = DATASET_TARGET_CONFIGS.get(normalized)
    if config is None:
        supported = ", ".join(sorted(DATASET_TARGET_CONFIGS))
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name!r}. Supported datasets: {supported}"
        )
    return config

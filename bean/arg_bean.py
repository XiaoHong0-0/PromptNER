from dataclasses import dataclass, field

import torch
import os


@dataclass
class BaseArgBean(object):
    """
    参数定义基类
    """
    # 数据文件相关
    train_data_path: str = field(default=None)
    dev_data_path: str = field(default=None)
    test_data_path: str = field(default=None)
    model_save_path: str = field(default=None)

    # 状态相关
    do_train: bool = field(default=False)
    do_predict: bool = field(default=False)

    # 模型参数相关
    dataloader_proc_num: int = field(default=4)
    epoch_num: int = field(default=5)
    eval_batch_step: int = field(default=2)
    learning_rate: float = field(default=0.2)
    max_input_len: int = field(default=32)
    output_dir: str = field(default=None)
    pad_to_max_length: bool = field(default=False)
    per_device_train_batch_size: int = field(default=128)
    per_device_eval_batch_size: int = field(default=128)
    pretrain_model_path: str = field(default=None)
    require_improvement_step: int = field(default=1000)
    seed: int = field(default=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def train_batch_size(self) -> int:
        """
        训练batch_size，多卡训练时为per_device_train_batch_size*device_count
        当前只考虑了GPU情况
        """
        train_batch_size = self.per_device_train_batch_size * max(1, torch.cuda.device_count())
        return train_batch_size

    @property
    def test_batch_size(self) -> int:
        """
        训练batch_size，多卡训练时为per_device_train_batch_size*device_count
        当前只考虑了GPU情况
        """
        test_batch_size = self.per_device_eval_batch_size * max(1, torch.cuda.device_count())
        return test_batch_size


@dataclass
class BertArgBean(BaseArgBean):
    """
    Bert 模型参数
    """
    pass


@dataclass
class T5ArgBean(BaseArgBean):
    """
    T5模型参数定义类
    """
    # 模型参数相关
    max_target_len: int = field(default=32)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    beam_num: int = field(default=1)
    ignore_pad_token_for_loss: bool = field(default=True)

    # 是否拼接上下文
    is_concat_context: bool = field(default=False)
    # 是否处理数据中全大写单词
    is_truecase_sent: bool = field(default=True)

    label_dict = {
        # DNRTI
        "HackOrg": "hacker organization",
        "OffAct": "attack",
        "SamFile": "sample file",
        "SecTeam": "security team",
        "Tool": "tool",
        "Time": "time",
        "Purp": "purpose",
        "Area": "area",
        "Idus": "industry",
        "Org": "organization",
        "Way": "way",
        "Exp": "loophole",
        "Features": "features",
        # ACE2004
        "GPE": "geo-political",
        "ORG": "organization",
        "PER": "person",
        "FAC": "facility",
        "VEH": "vehicle",
        "LOC": "location",
        "WEA": "weapon",
        # CADEC
        "ADR": "adverse drug reaction",
        # GENIA
        "cell_line": "cell line",
        "cell_type": "cell type",
        "DNA": "DNA",
        "RNA": "RNA",
        "protein": "protein",
        # Conll2003
        # "PER": "person",
        # "ORG": "organization",
        # "LOC": "location",
        "MISC": "miscellaneous",
        # Ontonotes-v5
        "CARDINAL": "cardinal",
        "DATE": "date",
        "EVENT": "event",
        # "FAC": "facility",
        # "GPE": "geo-political entity",
        "LANGUAGE": "language",
        "LAW": "law",
        # "LOC": "location",
        "MONEY": "money",
        "NORP": "affiliation",
        "ORDINAL": "ordinal",
        # "ORG": "organization",
        "PERCENT": "percent",
        "PERSON": "person",
        "PRODUCT": "product",
        "QUANTITY": "quantity",
        "TIME": "time",
        "WORK_OF_ART": "work of art"
    }

    label_set = set(label_dict.values())

    # 是否使用prompt
    prompt: bool = field(default=False)




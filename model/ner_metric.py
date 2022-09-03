from datasets import load_metric
from util.entity_util import EntityUtil


class NERMtric(object):
    """
    NER 模型处理类
    """
    def __init__(self, args):
        self.args = args
        self.seq_metric = load_metric("seqeval")
        self.static_dict = {"pred_num": 0, "label_num": 0, "pred_right_num": 0}

    def add_generate_batch(self, batch_pred_tokens=None, batch_label_tokens=None):
        """
        添加生成模型中每个batch的结果对
        :param batch_pred_tokens:
        :param batch_label_tokens:
        :return:
        """
        pred_num = 0
        label_num = 0
        pred_right_num = 0
        for pred_tokens, label_tokens in zip(batch_pred_tokens, batch_label_tokens):
            # print("pred_tokens: ", pred_tokens)
            # print("label_tokens: ", label_tokens)
            pred_entity_list = EntityUtil.get_generate_entity_by_mark(pred_tokens.split(), self.args.label_set)
            label_entity_list = EntityUtil.get_generate_entity_by_mark(label_tokens.split(), self.args.label_set)

            # print("pred_entity_list: ", len(pred_entity_list))
            # print("label_entity_list: ", len(label_entity_list))

            # token全部转为小写评测
            pred_entity_list = [[entity[0].lower(), entity[1].lower()] for entity in pred_entity_list]
            label_entity_list = [[entity[0].lower(), entity[1].lower()] for entity in label_entity_list]

            pred_num += len(pred_entity_list)
            label_num += len(label_entity_list)
            pred_right_num += len([pred_entity for pred_entity in pred_entity_list if pred_entity in label_entity_list])

        self.static_dict["pred_num"] = self.static_dict.get("pred_num", 0) + pred_num
        self.static_dict["label_num"] = self.static_dict.get("label_num", 0) + label_num
        self.static_dict["pred_right_num"] = self.static_dict.get("pred_right_num", 0) + pred_right_num

    def compute_generate_metric(self):
        """
        计算生成模型中所有batch的评测指标
        :return:
        """
        pred_num = self.static_dict.get("pred_num", 0)
        label_num = self.static_dict.get("label_num", 0)
        pred_right_num = self.static_dict.get("pred_right_num", 0)

        precision = 0 if pred_num == 0 else (pred_right_num / pred_num)
        recall = 0 if label_num == 0 else (pred_right_num / label_num)
        f1 = 0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)

        # 重置统计结果，防止多次验证时重复计算
        self.static_dict = {"pred_num": 0, "label_num": 0, "pred_right_num": 0}

        return {"precision": precision, "recall": recall, "f1": f1}

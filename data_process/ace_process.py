import sys

sys.path.append("../../MyPromptNER")

import json
from bean.arg_bean import T5ArgBean


class ACEProcess(object):
    """
    处理ACE2004数据集、ACE2005数据集
    """

    def __init__(self):
        self.ner_tag_list = ["GPE", "ORG", "PER", "FAC", "VEH", "LOC", "WEA"]

    def create_label(self, text, span_start_list, span_end_list):
        index_list = []
        label_list = []
        token_list = text.split()
        # print(token_list)
        label_index = 0
        label_dict = T5ArgBean.label_dict
        for label_span_start, label_span_end in zip(span_start_list, span_end_list):
            if len(label_span_start) == 0 and len(label_span_end) == 0:
                label_index += 1
                continue
            else:
                for start_index, end_index in zip(label_span_start, label_span_end):
                    index_list.append([start_index, end_index])
                    # [[4, 4], [0, 2]]
                    # [[0, 2], [1, 1], [4, 6], [5, 5], [1, 5], [1, 4], [1, 8], [2, 7]]
                    entity = token_list[start_index: end_index + 1]
                    label = self.ner_tag_list[label_index]
                    # entity.append('[SEP]')
                    entity.append(label_dict[label])
                    label_list.append(entity)
                label_index += 1
        return token_list, label_list, index_list

    def get_entity_list(self, text, span_start_list, span_end_list):
        token_list, label_list, index_list = self.create_label(text, span_start_list, span_end_list)
        entity_list = []
        for label, index in zip(label_list, index_list):
            if index[0] == index[1]:
                entity_list.append(token_list[index[0]])
            else:
                for i in range(index[0], index[1] + 1):
                    entity_list.append(token_list[i])
            entity_list.append(',')
            entity_list.append(label[-1])
            entity_list.append(';')

        return entity_list

    def format_file(self, read_file, write_file, datasets_type):
        with open(read_file, 'r') as f1, open(write_file, 'w+') as f2:
            load_dict = json.load(f1)
            # 7 label: {"GPE", "ORG" ,"PER", "FAC", "VEH", "LOC", "WEA"}
            total_num = len(load_dict)
            idx = 0
            single_data = {}
            span_start = []
            span_end = []
            for index in range(total_num):
                # span.append(load_dict[index]['span_position'])
                span_start.append(load_dict[index]['start_position'])
                span_end.append(load_dict[index]['end_position'])
                if (index + 1) % len(self.ner_tag_list) == 0:
                    idx += 1
                    single_data["id"] = "ace2004" + "-" + datasets_type + "-" + str(idx)
                    single_data["text"] = load_dict[index]['context']
                    single_data["token_list"] = load_dict[index]['context'].split()
                    single_data["entity_list"] = self.get_entity_list(single_data["text"], span_start, span_end)
                    span_start = []
                    span_end = []
                    f2.write(json.dumps(single_data) + '\n')


if __name__ == '__main__':
    read_train_data = "../data/datasets/ACE2004/origin/mrc-ner.train"
    read_dev_data = "../data/datasets/ACE2004/origin/mrc-ner.dev"
    read_test_data = "../data/datasets/ACE2004/origin/mrc-ner.test"

    write_train_data = "../data/datasets/ACE2004/format/train.txt"
    write_dev_data = "../data/datasets/ACE2004/format/valid.txt"
    write_test_data = "../data/datasets/ACE2004/format/test.txt"

    ace_process = ACEProcess()
    ace_process.format_file(read_train_data, write_train_data, "train")
    ace_process.format_file(read_dev_data, write_dev_data, "dev")
    ace_process.format_file(read_test_data, write_test_data, "test")

class EntityUtil(object):
    """
    实体处理相关工具类
    """

    @staticmethod
    def get_generate_entity_by_mark(generate_word_list: list, label_set: set) -> list:
        """
        根据生成模型结果获取实体列表
        :param generate_word_list: 生成的token序列
        :param label_set: 实体类型集合
        :return: ['Syria', ',', 'Location', ';', 'Lloyds', 'Shipping', ',', 'Organization']
         -> [['Syria', 'Location'], ['Lloyds Shipping', 'Organization']]
        """
        # print("label_set: ", label_set)
        # print("generate_word_list: ", generate_word_list)
        entity_list = []
        name_list = []
        for word in generate_word_list:
            if word not in label_set:
                name_list.append(word)
            else:
                entity_list.append([" ".join(name_list), word])
                name_list = []
        return entity_list

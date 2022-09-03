import re
import truecase


class TextUtil(object):
    """
    文本处理工具类
    """

    @staticmethod
    def truecase_sentence(token_list):
        """
        将全大写英文字符转化为首字母大写的字符
        :param token_list:
        :return:
        """
        new_token_list = token_list[:]
        # 仅转换均为英文的token
        en_idx_token_list = [(idx, token) for idx, token in enumerate(token_list) if all(c.isalpha() for c in token)]
        en_token_list = [token for _, token in en_idx_token_list if re.match(r'\b[A-Z\.\-]+\b', token)]

        if len(en_token_list) and len(en_token_list) == len(en_idx_token_list):
            case_token_list = truecase.get_true_case(' '.join(en_token_list)).split()

            if len(case_token_list) == len(en_token_list):
                for (idx, token), case_token in zip(en_idx_token_list, case_token_list):
                    new_token_list[idx] = case_token
                return new_token_list

        return token_list




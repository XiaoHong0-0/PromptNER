import logging


class LogUtil(object):
    """
    日志工具类
    """
    # 日志格式
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(filename)s line:%(lineno)d] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()


if __name__ == '__main__':
    # from util.log_util import LogUtil
    logger = LogUtil.logger
    # print -> logger.info()
    # print("hello world")
    logger.info("hello world!")
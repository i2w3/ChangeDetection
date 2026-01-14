import sys
import time
import logging
from pathlib import Path
from typing import Optional


class Logger:
    def __init__(self, logger_name:str, init_logger:bool = False) -> None:
        '''等待调用 self.init_logger() 初始化
        '''
        self.logger_name = logger_name
        self.logger_init = init_logger
        if init_logger:
            self.init_logger()

    def init_logger(self, logger_path:Optional[Path] = None, console_logging:bool = True) -> None:
        ''' 初始化 logger
        '''
        local_time = time.strftime("%Y-%m-%dT%H%M%S", time.localtime())
        self.logger_path = logger_path if logger_path is not None else Path(f"./logs/{local_time}/{local_time}.log")
        Path.mkdir(self.logger_path.parent, parents=True, exist_ok=True)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 初始化 logger
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)
        # 文件日志
        fh = logging.FileHandler(self.logger_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        # 控制台日志
        if console_logging:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        self.console_logging = console_logging
        self.logger_init = True


    def __call__(self, level:str, note:str) -> None:
        ''' log 一条日志
        '''
        if not self.logger_init:
            self.init_logger()
            self.logger_init = True
        if level == 'debug':
            self.logger.debug(note)
        elif level == 'info':
            self.logger.info(note)
        elif level == 'warning':
            self.logger.warning(note)
        elif level == 'error':
            self.logger.error(note)
        else:
            self.logger.error("didn't use the true level, but log: " + note)

    
logger = Logger(Path(sys.argv[0]).stem)

if __name__ == "__main__":
    logger = Logger("test_logger")
    logger('debug','Debug message')
    logger('info','Info message')
    logger('warning','Warning message')
    logger('error','Error message')

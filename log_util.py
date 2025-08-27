#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created: 2025-07-18
import os
import sys
import time


try:
    import colorama
    colorama.init(autoreset=True)

    class Fore:
        Black = colorama.Fore.LIGHTBLACK_EX
        Blue = colorama.Fore.LIGHTBLUE_EX
        Cyan = colorama.Fore.LIGHTCYAN_EX
        Green = colorama.Fore.LIGHTGREEN_EX
        Magenta = colorama.Fore.LIGHTMAGENTA_EX
        Red = colorama.Fore.LIGHTRED_EX
        White = colorama.Fore.LIGHTWHITE_EX
        Yellow = colorama.Fore.LIGHTYELLOW_EX

        DarkBlack = colorama.Fore.BLACK
        DarkBlue = colorama.Fore.BLUE
        DarkCyan = colorama.Fore.CYAN
        DarkGreen = colorama.Fore.GREEN
        DarkMagenta = colorama.Fore.MAGENTA
        DarkRed = colorama.Fore.RED
        DarkWhite = colorama.Fore.WHITE
        DarkYellow = colorama.Fore.YELLOW

        Reset = colorama.Fore.RESET

    class Back:
        Black = colorama.Back.LIGHTBLACK_EX
        Blue = colorama.Back.LIGHTBLUE_EX
        Cyan = colorama.Back.LIGHTCYAN_EX
        Green = colorama.Back.LIGHTGREEN_EX
        Magenta = colorama.Back.LIGHTMAGENTA_EX
        Red = colorama.Back.LIGHTRED_EX
        White = colorama.Back.LIGHTWHITE_EX
        Yellow = colorama.Back.LIGHTYELLOW_EX

        DarkBlack = colorama.Back.BLACK
        DarkBlue = colorama.Back.BLUE
        DarkCyan = colorama.Back.CYAN
        DarkGreen = colorama.Back.GREEN
        DarkMagenta = colorama.Back.MAGENTA
        DarkRed = colorama.Back.RED
        DarkWhite = colorama.Back.WHITE
        DarkYellow = colorama.Back.YELLOW

        Reset = colorama.Back.RESET

    class Style:
        Bright = colorama.Style.BRIGHT
        Dim = colorama.Style.DIM
        Normal = colorama.Style.NORMAL
        Reset = colorama.Style.RESET_ALL

except Exception:
    # Windows 7 console doesn't support colors
    if sys.platform == 'win32': # works on Windows 10
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)

    class Fore:
        Black = "\033[90m"
        Red = "\033[91m"
        Green = "\033[92m"
        Yellow = "\033[93m"
        Blue = "\033[94m"
        Magenta = "\033[95m"
        Cyan = "\033[96m"
        White = "\033[97m"

        DarkBlack = "\033[30m"
        DarkRed = "\033[31m"
        DarkGreen = "\033[32m"
        DarkYellow = "\033[33m"
        DarkBlue = "\033[34m"
        DarkMagenta = "\033[35m"
        DarkCyan = "\033[36m"
        DarkWhite = "\033[37m"

        Reset = "\033[39m"

    class Back:
        Black = "\033[100m"
        Red = "\033[101m"
        Green = "\033[102m"
        Yellow = "\033[103m"
        Blue = "\033[104m"
        Magenta = "\033[105m"
        Cyan = "\033[106m"
        White = "\033[107m"

        DarkBlack = "\033[40m"
        DarkRed = "\033[41m"
        DarkGreen = "\033[42m"
        DarkYellow = "\033[43m"
        DarkBlue = "\033[44m"
        DarkMagenta = "\033[45m"
        DarkCyan = "\033[46m"
        DarkWhite = "\033[47m"

        Reset = "\033[49m"

    class Style:
        Bright = ''
        Dim = ''
        Normal = ''
        Reset = ''


try:
    raise ImportError('use logging first')

    from loguru import logger

    def config_logger(logger, log_level = 'info', log_dir = 'logs', log_file = 'app.log',
                      backup_count = 15, log_to_stdout = True):
        if log_dir and log_dir != '.':
            os.makedirs(log_dir, exist_ok=True)
        log_level = log_level.upper()
        logger.remove()   # remove the default sink of sys.stdout
        '''
        {
            'elapsed': datetime.timedelta(microseconds=30002),
            'exception': None,
            'extra': {},
            'file': (name='log_util.py', path='E:\\codes\\python\\automation\\log_util.py'),
            'function': 'log_test',
            'level': (name='CRITICAL', no=50, icon='☠️'),
            'line': 123,
            'message': 'hello world',
            'module': 'log_util',
            'name': '__main__',
            'process': (id=23244, name='MainProcess'),
            'thread': (id=25388, name='MainThread'),
            'time': datetime(2025, 8, 26, 10, 45, 45, 783528, tzinfo=datetime.timezone(datetime.timedelta(seconds=28800), '中国标准时间'))
        }
        '''
        file_format = '{time:YYYY-MM-DD HH:mm:ss.SSS} {level} T{thread} L{line} {function}: {message}'
        if log_to_stdout:
            console_format = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <lvl>{level}</lvl> '
                '{file},{line} <light-blue>T{thread}</light-blue> <light-cyan>{function}</light-cyan>'
                ': <lvl>{message}</lvl>')
            _stdout_logger_id = logger.add(sys.stdout, level=log_level, colorize=True, format=console_format)
        logger.add(f'{log_dir}/{log_file}', level=log_level, enqueue=True, rotation=f'00:00:00',
                   retention=f'{backup_count} days', compression='zip', format=file_format)

except ImportError:

    import zipfile
    import logging
    import logging.handlers

    logger = logging.getLogger()


    class LogFormatter(logging.Formatter):
        default_time_format = '%Y-%m-%d %H:%M:%S'

        def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
            super(LogFormatter, self).__init__(fmt, datefmt, style, validate)


    class ZipTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
        def doRollover(self):
            if self.stream:
                self.stream.close()
                self.stream = None

            last_rollover_time_tuple = time.localtime(self.rolloverAt - self.interval)
            old_log_suffix = time.strftime(self.suffix, last_rollover_time_tuple)
            old_log_path = f"{self.baseFilename}.{old_log_suffix}"

            super().doRollover()

            if os.path.exists(old_log_path):
                zip_filename = f"{old_log_path}.zip"
                try:
                    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.write(old_log_path, os.path.basename(old_log_path))
                    print(f"compress: {old_log_path} -> {zip_filename}")
                    os.remove(old_log_path)
                    print(f"remove: {old_log_path}")
                except Exception as ex:
                    print(f"compress failed: {old_log_path}: {ex!r}")
            else:
                print(f"can not find: {old_log_path}")


    def config_logger(logger: logging.Logger, log_level = 'info', log_dir = 'logs', log_file = 'app.log',
                      backup_count = 15, log_to_stdout = True):
        if log_dir and log_dir != '.':
            os.makedirs(log_dir, exist_ok=True)
        log_level = logging._nameToLevel[log_level.upper()]
        logging.Formatter.default_msec_format = '%s.%03d'
        file_formatter = LogFormatter('%(asctime)s %(levelname)s T%(thread)d L%(lineno)d %(funcName)s: %(message)s')
        logger.setLevel(log_level)
        file_handler = ZipTimedRotatingFileHandler(
            f'{log_dir}/{log_file}',
            when="midnight",    # midnight
            interval=1,         # every day
            backupCount=backup_count
        )
        file_handler.setFormatter(file_formatter)
        if log_to_stdout:
            logging.basicConfig(
                level=log_level,
                format=f'%(asctime)s %(levelname)s %(filename)s,%(lineno)d {Fore.Blue}T%(thread)d{Fore.Reset} {Fore.Cyan}%(funcName)s{Fore.Reset}: %(message)s',
                # format=f'%(asctime)s %(levelname)s %(filename)s,%(lineno)d T%(thread)d %(funcName)s: %(message)s', # no color
            )
        else:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        logger.addHandler(file_handler)


def get_uvicorn_logging_config() -> dict:
    '''
    config = get_uvicorn_logging_config()
    uvicorn.run(app, lifespan='on', host=host, port=port, workers=1, log_level='info', log_config=config)
    '''
    try:
        from uvicorn.config import LOGGING_CONFIG

        LOGGING_CONFIG["formatters"]["default"]["fmt"] = '%(asctime)s.%(msecs)03d %(levelprefix)s %(message)s'
        LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s.%(msecs)03d %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
        LOGGING_CONFIG["formatters"]["default"]["datefmt"] = '%Y-%m-%d %H:%M:%S'
        LOGGING_CONFIG["formatters"]["access"]["datefmt"] = '%Y-%m-%d %H:%M:%S'
        return LOGGING_CONFIG
    except ImportError:
        return None


if __name__ == '__main__':
    def log_test():
        logger.info('hello world')
        logger.warning('hello world')
        logger.error('hello world')
        logger.critical('hello world')

    config_logger(logger, log_to_stdout=True)
    logger.debug('hello world')
    log_test()
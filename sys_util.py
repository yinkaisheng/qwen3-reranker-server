#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import traceback
import subprocess

from log_util import logger, Fore


def get_exception_stack(with_color: bool = False) -> str:
    ex_type, ex_value, ex_traceback = sys.exc_info()
    # printx(ex_type, ex_value, ex_traceback)
    tb: traceback.StackSummary = traceback.extract_tb(ex_traceback)
    # printx(tb)
    frame: traceback.FrameSummary
    frame_count = len(tb)
    color = Fore.Red if with_color else ''
    stacks = [f'----{ex_type}']
    for i, frame in enumerate(tb):
        # filename, lineno, funcname, code = frame
        filename = frame.filename[frame.filename.rfind(os.sep)+1:]
        if i < frame_count - 1:
            stacks.append(f'|{"  "*i}{filename} {frame.lineno} {frame.name}: {frame.line}')
        else:
            stacks.append(f'|{color}{"  "*i}{filename} {frame.lineno} {frame.name}: {frame.line}')
    stacks.append(f'{Fore.Reset}----' if with_color else '----')
    return '\n'.join(stacks)


def run_cmd(cmd, text: bool=True, encoding: str='utf-8', shell: bool=True,
            env: dict=None, timeout: int=10) -> dict:
    result = {}
    logger.info(f'run cmd={cmd!r}')
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=text, encoding=encoding, shell=shell, env=env) as process:
        try:
            stdout, stderr = process.communicate(input=None, timeout=timeout)
            result['stdout'] = stdout
            result['stderr'] = stderr
        except subprocess.TimeoutExpired as ex:
            process.kill()
            if subprocess._mswindows:
                ex.stdout, ex.stderr = process.communicate()
            else:
                process.wait()
            result['timeout'] = True
        except Exception as ex:  # Including KeyboardInterrupt, communicate handled that.
            process.kill()
            result['exception'] = repr(ex)
        result['exit_code'] = process.poll()
    return result

from pathlib import Path
import os
from datetime import datetime
import json


class Logger(object):

    def __init__(self, base_dir):
        self.ppath = os.path.join(base_dir, 'progress.json')
        self.lpath = os.path.join(base_dir, 'log.txt')

        # check file
        self._is_exist(self.ppath)
        self._is_exist(self.lpath)
        # create file
        self._touch(self.ppath)
        self._touch(self.lpath)

    def _is_exist(self, fpath):
        if os.path.exists(fpath):
            print('{} already exists!'.format(fpath))
            b = int(input('do you wanna overwrite ? (0 or 1): '))
            if not b:
                raise Exception

    def _touch(self, fpath):
        Path(fpath).touch()

    def dump(self, dict_):
        json_str = json.dumps(dict_)
        print(json_str)
        json_str += '\n'
        self._write_add(self.ppath, json_str)

    def log(self, text):
        now = datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        out = '[{}] {}\n'.format(now_str, text)
        print(out.strip())
        self._write_add(self.lpath, out)

    def _write_add(self, fpath, s):
        with open(fpath, 'a') as f:
            f.write(s)

import regex as re
import json
import arrow
from Parser import Parser


def init():
    with open('./resources/regex.txt', encoding='utf-8') as f:
        _regex = f.read()
    _pattern = re.compile(_regex)
    _holiday = json.load(open('./resources/holiday.json', 'r', encoding='utf-8'))
    return _pattern, _holiday


class TimeResolver:
    def __init__(self):
        self.pattern, self.holiday = init()

    def parse(self, sequence, base_time=arrow.now(), prefer_future=False):
        return Parser(self.pattern, sequence, base_time, self.holiday, prefer_future)

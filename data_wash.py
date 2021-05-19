import json
import os

import bs4
import regex as re
from bs4 import BeautifulSoup

_dir = r'D:\dataset\tempeval-training-2\chinese\html'
_dict = r'D:\dataset\tempeval-training-2\chinese\data\dct.txt'

tag_pattern = re.compile(r'[t|e]\d{1,3}')
text_pattern = re.compile(r'\[.*?\]')

_base_time_dict = {}


def dct():
    _f = open(_dict, 'r')
    for _line in _f.readlines():
        _key = _line[0:9]
        _base_time = _line[20:30]
        _base_time_dict[_key] = _base_time
    return _base_time_dict


def find(_file):
    print(_file + '-----------------')
    f = open(os.sep.join([_dir, _file]))
    soup = BeautifulSoup(f, "html.parser")
    dic1 = {}
    dic2 = {}
    lines = []
    dicts = {}
    _id = _file.replace('.html', '')
    
    for _tab in soup.find_all('div', 'sentence'):
        _p = _tab.find_all('p', 'line')[0]
        
        _line = _p.text
        _line = _line.replace(' ', '')
        
        for _t, _tag in zip(text_pattern.findall(_line), tag_pattern.findall(_line)):
            if _tag.startswith('t'):
                dic1[_tag] = {'text': _t[1: -1]}
        
        _attrs = _tab.find_all('p', 'attributes')
        for _attr in _attrs:
            _text = _attr.text
            if _text.startswith('t'):
                _text = _text.split(' ')
                _label = _text[0]
                _type = _text[2].split(':')[1]
                _val = None
                if len(_text) > 4:
                    _val = _text[3].split(':')[1]
                dic2[_label] = {'type': _type, 'value': _val}
        
        for _t in _p:
            if type(_t) is bs4.element.Tag:
                for _tt in _t:
                    if _tt.name == 'sup':
                        _tt.clear()
            if _t.name == 'sup':
                _t.clear()
        _line = _p.text
        _line = _line.replace(' ', '')
        _line = _line.replace('[', '')
        _line = _line.replace(']', '')
        
        lines.append(_line)
    
    for k, v in dic1.items():
        v.update(dic2[k])
        dicts[k] = v
    
    return {'id': _id, 'base_time': _base_time_dict[_id], 'content': {'label': dicts, 'text': lines[:-1]}}


if __name__ == '__main__':
    _list = []
    dct()
    
    for file in os.listdir(_dir):
        if file.startswith('chtb'):
            _list.append(find(file))
    json.dump(_list, open(os.sep.join([_dir, 'time_annotation.json']), 'w', encoding='utf-8'), ensure_ascii=False)

import arrow
import json
from TimeResolver import TimeResolver
from dateutil import tz

_data_file = r'./resources/time_annotation.json'
_data = json.load(open(_data_file, 'r', encoding='utf-8'))

if __name__ == '__main__':
    tr = TimeResolver()
    
    for _article in _data:
        _id = _article.get('id')
        # if _id == 'chtb_0031':
        p = tr.parse(''.join(_article.get('content').get('text')),
                     base_time=arrow.get(_article.get('base_time'), 'YYYY-MM-DD', tzinfo=tz.tzlocal()))
        p.find()

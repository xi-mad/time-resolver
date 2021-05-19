import json

_data_file = r'./resources/time_annotation.json'
_data = json.load(open(_data_file, 'r', encoding='utf-8'))

labels = [len(_e['content']['label']) for _e in _data]

time_point_count = 0
for _e in _data:
    for _val in _e['content']['label'].values():
        if _val['type'] == 'DATE' and _val['value'] is not None and _val['value'] != '' and '0' <= _val['value'][0] <= '9':
            time_point_count += 1

print('文章数：', len(_data))  # 44
print('时间个数:', sum(labels))  # 766
print('时间点个数:', time_point_count)  # 403



import calendar
import pickle

import arrow
import regex as re
from tensorflow import keras

model = keras.models.load_model('./resources/judge_model')

with open('./resources/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# import logging
# logging.getLogger().setLevel(logging.DEBUG)


class StringUtil:
    @staticmethod
    def delete_words_by_rules(target, rules):
        pattern = re.compile(rules)
        return pattern.sub('', target)

    @classmethod
    def number_translate(cls, target):
        """
        该方法可以将字符串中所有的用汉字表示的数字转化为用阿拉伯数字表示的数字
        如"这里有一千两百个人，六百零五个来自中国"可以转化为
        "这里有1200个人，605个来自中国"
        此外添加支持了部分不规则表达方法
        如两万零六百五可转化为20650
        两百一十四和两百十四都可以转化为214
        一六零加一五八可以转化为160+158
        该方法目前支持的正确转化范围是0-99999999
        该功能模块具有良好的复用性
        :param target: 待转化的字符串
        :return: 转化完毕后的字符串
        """
        pattern = re.compile(u"[一二两三四五六七八九123456789]万[一二两三四五六七八九123456789](?!([千百十]))")
        match = pattern.finditer(target)
        for m in match:
            group = m.group()
            s = group.split(u"万")
            s = list(filter(None, s))
            num = 0
            if len(s) == 2:
                num += cls.word_2_num(s[0]) * 10000 + cls.word_2_num(s[1]) * 1000
            target = pattern.sub(str(num), target, 1)

        pattern = re.compile(u"[一二两三四五六七八九123456789]千[一二两三四五六七八九123456789](?!([百十]))")
        match = pattern.finditer(target)
        for m in match:
            group = m.group()
            s = group.split(u"千")
            s = list(filter(None, s))
            num = 0
            if len(s) == 2:
                num += cls.word_2_num(s[0]) * 1000 + cls.word_2_num(s[1]) * 100
            target = pattern.sub(str(num), target, 1)

        pattern = re.compile(u"[一二两三四五六七八九123456789]百[一二两三四五六七八九123456789](?!十)")
        match = pattern.finditer(target)
        for m in match:
            group = m.group()
            s = group.split(u"百")
            s = list(filter(None, s))
            num = 0
            if len(s) == 2:
                num += cls.word_2_num(s[0]) * 100 + cls.word_2_num(s[1]) * 10
            target = pattern.sub(str(num), target, 1)

        pattern = re.compile(u"[零〇一二两三四五六七八九]")
        match = pattern.finditer(target)
        for m in match:
            target = pattern.sub(str(cls.word_2_num(m.group())), target, 1)

        # pattern = re.compile(u"(?<=(周|星期))[末天日]")
        # match = pattern.finditer(target)
        # for m in match:
        #     target = pattern.sub(str(cls.word_2_num(m.group())), target, 1)

        pattern = re.compile(u"(?<!(周|星期))0?[0-9]?十[0-9]?")
        match = pattern.finditer(target)
        for m in match:
            group = m.group()
            s = group.split(u"十")
            ten = cls.int(s[0])
            if ten == 0:
                ten = 1
            unit = cls.int(s[1])
            num = ten * 10 + unit
            target = pattern.sub(str(num), target, 1)

        pattern = re.compile(u"0?[1-9]百[0-9]?[0-9]?")
        match = pattern.finditer(target)
        for m in match:
            group = m.group()
            s = group.split(u"百")
            s = list(filter(None, s))
            num = 0
            if len(s) == 1:
                hundred = int(s[0])
                num += hundred * 100
            elif len(s) == 2:
                hundred = int(s[0])
                num += hundred * 100
                num += int(s[1])
            target = pattern.sub(str(num), target, 1)

        pattern = re.compile(u"0?[1-9]千[0-9]?[0-9]?[0-9]?")
        match = pattern.finditer(target)
        for m in match:
            group = m.group()
            s = group.split(u"千")
            s = list(filter(None, s))
            num = 0
            if len(s) == 1:
                thousand = int(s[0])
                num += thousand * 1000
            elif len(s) == 2:
                thousand = int(s[0])
                num += thousand * 1000
                num += int(s[1])
            target = pattern.sub(str(num), target, 1)

        pattern = re.compile(u"[0-9]+万[0-9]?[0-9]?[0-9]?[0-9]?")
        match = pattern.finditer(target)
        for m in match:
            group = m.group()
            s = group.split(u"万")
            s = list(filter(None, s))
            num = 0
            if len(s) == 1:
                num += int(s[0]) * 10000
            elif len(s) == 2:
                num += int(s[0]) * 10000
                num += int(s[1])
            target = pattern.sub(str(num), target, 1)

        return target

    @staticmethod
    def word_2_num(s):
        _map = {
            '零': 0,
            '0': 0,
            '〇': 0,
            '一': 1,
            '1': 1,
            '二': 2,
            '两': 2,
            '2': 2,
            '三': 3,
            '3': 3,
            '四': 4,
            '4': 4,
            '五': 5,
            '5': 5,
            '六': 6,
            '6': 6,
            '七': 7,
            '7': 7,
            # 天？日？末？
            '八': 8,
            '8': 8,
            '九': 9,
            '9': 9
        }
        if s in _map.keys():
            return _map.get(s)
        return -1

    @staticmethod
    def int(s):
        try:
            return int(s)
        except:
            return 0


class Time:
    def __init__(self, _tu):
        self.base_time = _tu['context_time']
        _tu['context_time'] = self.base_time.format('YYYY-MM-DD HH:mm:ss')
        self.time = self.base_time
        self.time_unit = _tu
        self.smallest_unit = -1

        self.terms = ['c', 'dy', 'y', 'yn', 'M', 'S', 'w', 'd', 'h', 'm', 's']
        self.ops = {'yc', 'yy', 'yM', 'yw', 'yd', 'yh', 'ym', 'ys'}

        if self.time_unit['ref'] != '':
            return
        
        self.set_time()
        self.set_offset()

    def set(self, op, val):
        if op in self.terms:
            self._set_terms(op, val)
        if op in self.ops:
            self._set_ops(op, val)

    def _set_terms(self, op, val):
        if 'c' == op:
            self.time = self.time.replace(year=(val - 1) * 100)
        if 'dy' == op or 'yn' == op:
            self.time = self.time.replace(year=self.time.year // 100 * 100 + val)
        if 'y' == op:
            self.time = self.time.replace(year=val)
        if 'M' == op:
            _day_of_month = Parser.month_last_day(self.time.year, val)
            self.time = self.time.replace(month=val, day=_day_of_month)
        if 'S' == op:
            _month = (val - 1) * 3 + 1
            _day_of_month = Parser.month_last_day(self.time.year, _month)
            self.time = self.time.replace(month=_month, day=_day_of_month)
        if 'w' == op:
            _week_day = self.time.weekday() + 1
            _day_offset = val - _week_day
            self.time = self.time.shift(days=_day_offset)
        if 'd' == op:
            _day_of_month = Parser.month_last_day(self.time.year, self.time.month)
            self.time = self.time.replace(day=min(val, _day_of_month))
        if 'h' == op:
            self.time = self.time.replace(hour=val)
        if 'm' == op:
            self.time = self.time.replace(minute=val)
        if 's' == op:
            self.time = self.time.replace(second=val)

    def _set_ops(self, op, val):
        if 'yc' == op:
            _year = self.time.year + (val * 100)
            self.time = self.time.replace(year=_year)
        if 'yy' == op:
            _year = self.time.year + val
            self.time = self.time.replace(year=_year)
        if 'yM' == op:
            self.time = self.time.shift(months=val)
        if 'yw' == op:
            self.time = self.time.shift(weeks=val)
        if 'yd' == op:
            self.time = self.time.shift(days=val)
        if 'yh' == op:
            self.time = self.time.shift(hours=val)
        if 'ym' == op:
            self.time = self.time.shift(minutes=val)
        if 'ys' == op:
            self.time = self.time.shift(seconds=val)

    def set_time(self):
        self._set_list(self.time_unit['time'])
        
    def set_offset(self):
        for _k, _v in self.time_unit['offset'].items():
            self.set(_k, _v)
        
    def _set_list(self, _ls):
        for _t in _ls:
            for _k, _v in _t.items():
                self.set(_k, _v)

    def get_time(self):
        if self.time_unit['ref'] != '':
            return self.time_unit['ref']
    
        _times = [str(i) for i in (self.time.year, self.time.month, self.time.day,
                  self.time.hour, self.time.minute, self.time.second)][:self.smallest_unit + 1]
        for _i in range(len(_times)):
            if len(_times[_i]) == 1:
                _times[_i] = '0' + _times[_i]
                
        if self.smallest_unit < 3:
            return '-'.join(_times)
        else:
            return 'T'.join(['-'.join(_times[:3]), ':'.join(_times[3:])])

    def get_complete_time(self):
        return self.time.format()

    def find_smallest_unit(self):
        # range 0-5
        _map = {'c': 0, 'y': 0, 'n': 0, 'M': 1, 'S': 1, 'w': 2, 'd': 2, 'h': 3, 'm': 4, 's': 5}
        for _tp in self.time_unit['time']:
            for _k, _v in _tp.items():
                self.smallest_unit = max(self.smallest_unit, _map.get(_k[-1]))
        for _k, _v in self.time_unit['offset'].items():
            self.smallest_unit = max(self.smallest_unit, _map.get(_k[-1]))

    def is_good_time(self):
        if self.time_unit['ref'] != '':
            return True
        self.find_smallest_unit()
        return True if self.smallest_unit > -1 else False


# 计算时间是未来还是过去
class Parser:
    def __init__(self, pattern, sequence, base_time, holiday, prefer_future):
        self.pattern = pattern
        self.sequence = sequence
        self.regular_sequence = self.prefilter(self.sequence)
        self.base_time = base_time
        self.holiday = holiday
        self.prefer_future = prefer_future
        self.time_sequence = []
        # c(世纪),dy(年代),y,yn(年的后两位),M,S(季度),d,h,m,s | yc,yy,yM,yw,yd,yh,ym,ys
        self.time_unit = []
        self.time = []

    def match(self):
        _end = -1
        _pointer = 0

        for _m in self.pattern.finditer(self.regular_sequence):
            _group = _m.group()
            _start = _m.start()
            if self.concat(_end, _start):
                _pointer -= 1
                self.time_sequence[_pointer]['text'].append(_group)
                self.time_sequence[_pointer]['end'] = _m.end()
            else:
                self.time_sequence.append({'text': [_group], 'start': _start, 'end': _m.end()})

            _end = _m.end()
            _pointer += 1

    def resolve(self):
        for _ts in self.time_sequence:
            self._resolve(_ts)

        self.time_unit = [_tu for _tu in self.time_unit if self.is_time(_tu) > 0.5]

        for _i in range(len(self.time_unit)):
            _tu = self.time_unit[_i]
            _context_time = self.find_context_time(_i)
            _tu['context_time'] = _context_time
            _time = Time(_tu)
            if _time.is_good_time():
                _tu['final'] = _time.get_time()
                _tu['final_complete'] = _time.get_complete_time()
        for _tu in self.time_unit:
            print(_tu['text'], '-----', end='')
            print(_tu['final'])
        print('-'*10)

    @staticmethod
    def month_last_day(y, m):
        _m_day = {
            1: 31, 3: 31, 5: 31, 7: 31, 8: 31, 10: 31, 12: 31,
            4: 30, 6: 30, 9: 30, 11: 30,
        }
        if m in _m_day.keys():
            return _m_day[m]
        return 29 if calendar.isleap(y) else 28

    def _resolve(self, ts):
        _time_list = []
        _offset_list = {}
        _holiday_pattern = re.compile(
            r'(中秋|清明|端午|教师|春|儿童|元旦|妇女|光棍|愚人|植树|父亲|母亲|国庆|劳动|圣诞)(节)|(7夕|立春|雨水|惊蛰|春分|清明|谷雨|立夏|小满|芒种|夏至|小暑|大暑|立秋|处暑|白露|秋分|寒露|霜降|立冬|小雪|大雪|冬至|小寒|大寒)')
        _century_pattern = re.compile('[1-9]?[0-9](?=世纪(初|末)?)')
        _century_offset_pattern = re.compile('(本|上|下|新|旧)世纪(初|末)?')

        _year_pattern = re.compile(r'([1-9][0-9]{3})(?=年(初|末|底)?)')
        _year_pattern_1 = re.compile(r'\d{4}(?=[至和])')
        _decade_pattern = re.compile('([1-9]0)(?=年代(初|末|中期|后期)?)')
        _year_offset_pattern = re.compile('(今|当|明|来|后|去|上|前|大前|这1)年(初|末|底)?')
        _year_offset_pattern1 = re.compile(r'\d+(?=(年(前|后)))')
        _season_pattern = re.compile('第?[1-4]季度')
        # todo: ys: 本 上 下季度 初末
        _month_pattern = re.compile('((10|11|12)|(0[1-9])|([1-9]))(?=月(份|初|末|底)?)')
        _month_offset_pattern = re.compile('(上|上上|下|下下|这个|本|次)(个)?月(初|末|底)?')
        _day_pattern = re.compile('((30|31)|([0-2][0-9])|([1-9]))(?=(日|号))')
        _day_pattern_1 = re.compile('(上旬|中旬|下旬)')
        _day_pattern_1_mapping = {
            '上旬': {'d': 5},
            '中旬': {'d': 15},
            '下旬': {'d': 25}
        }
        _week_pattern = re.compile('(上个|上上个|下个|下下个|这个|这|本|上|上上|下|下下)?((星期|周|礼拜)[1-6日天未])')
        _day_offset_pattern = re.compile('(今天|昨天|明天|前天|大前天|后天|大后天|今日|昨日|前日|明日|前1天)')
        _hour_pattern = re.compile('(?<=(早|晚|当晚)?)(([0-1][0-9])|([0-9])|(20|21|22|23))(?=(点|时)半?)')
        _hour_pattern_1 = re.compile('(早上|早晨|上午|中午|下午|傍晚|晚上|夜晚|今晚|午夜|凌晨)')
        _hour_pattern_1_mapping = {
            '早上': {'h': 10},
            '早晨': {'h': 8},
            '上午': {'h': 10},
            '中午': {'h': 12},
            '下午': {'h': 14},
            '傍晚': {'h': 18},
            '晚上': {'h': 20},
            '今晚': {'h': 20},
            '夜晚': {'h': 22},
            '午夜': {'h': 23},
            '凌晨': {'h': 0}
        }
        _minute_pattern = re.compile('(([0-5][0-9])|([0-9]))(?=(分钟前|分钟后|分钟之前|分钟之后|分钟|分前|分后|分之前|分之后|分))')
        _second_pattern = re.compile('([1-9]|[1-5][0-9])(?=(秒钟后|秒钟前|秒钟之后|秒钟之前|秒钟|秒前|秒后|秒之前|秒之后|秒))')
        
        _past_ref_pattern = re.compile(r'((近(几|\d+)?多?年(来|间)?)|过去((的?几10|的?\d{1,3})年间?)?|近(1段时)?期|几年|以[往|前]|最近(\d+年来)?|前几年|新近|近[日|期|来]|[当|那]时|(前)?不久|昔日|这期间|前期|晚期|最新|先前)')
        _present_ref_pattern = re.compile('(当[今|下|]|[当|目|日]前|如今|现[在|今]|与此同时)')
        _future_ref_pattern = re.compile(r'(未来|[今|以]后(\d+年)?)')
        _duration_pattern = re.compile(r'((\d+)?多年|半个世纪|[几|数]代人|半月(以上)?|[几|数][十|百|千|万|十万]?年|连(续多)?年|\d+年多|全年|\d+个月|长期|1直|(\d+)个?(年|学年|学期|季度|周|星期|天|小时|分钟|秒))')
        _cycle_pattern = re.compile(r'(每(\d+)?(年|学年|学期|季度|月|周|星期|日|天|小时|分钟|秒))')
        _dynasty_pattern = re.compile(r'(夏|商|西周|东周|春秋|战国|秦|西汉|东汉|西晋|东晋|南北朝|隋|唐|五代十国|宋|北宋|南宋|辽|西夏|明|清)')
        _zodiac_pattern = re.compile('((鼠|牛|虎|兔|龙|蛇|马|羊|猴|鸡|狗|猪)年)')
        _now = self.base_time
        _pm_flag = False
        _ref = ''
        for _t in ts['text']:
            _zodiac_match = _zodiac_pattern.search(_t)
            if _zodiac_match is not None:
                _ref = 'zodiac'
                continue
            _past_ref_match = _past_ref_pattern.search(_t)
            if _past_ref_match is not None:
                _ref = 'past'
                continue
            _present_ref_match = _present_ref_pattern.search(_t)
            if _present_ref_match is not None:
                _ref = 'present'
                continue
            _future_ref_match = _future_ref_pattern.search(_t)
            if _future_ref_match is not None:
                _ref = 'future'
                continue
            _cycle_match = _cycle_pattern.search(_t)
            if _cycle_match is not None:
                _ref = 'cycle'
                continue
            
            if _t == '上半年':
                _time_list.append({'M': 1})
                continue
            if _t == '下半年':
                _time_list.append({'M': 7})
                continue

            if _t == '1刻':
                _time_list.append({'m': 15})
                continue
            
            _holiday_match = _holiday_pattern.search(_t)
            if _holiday_match is not None:
                _g = _holiday_match.group()
                _value = self.holiday[_g]
                if _value == '':
                    # 阴历转阳历
                    pass
                else:
                    _month, _day = _value.split('-')
                    _time_list.extend([{'M': int(_month)}, {'d': int(_day)}])
                continue

            _century_match = _century_pattern.search(_t)
            if _century_match is not None:
                _g = _century_match.group()
                _time_list.append({'c': int(_g)})
                if _t[-1] == '末':
                    _time_list.append({'yn': 99})
                elif _t[-1] == '初':
                    _time_list.append({'yn': 0})
                continue

            _century_offset_match = _century_offset_pattern.search(_t)
            if _century_offset_match is not None:
                if _t[0] == '上' or _t[0] == '旧':
                    _offset_list.update({'yc': -1})
                elif _t[0] == '下' or _t[0] == '新':
                    _offset_list.update({'yc': 1})
                elif _t[0] == '本':
                    _offset_list.update({'yy': 0})

                if _t[-1] == '末':
                    _time_list.append({'yn': 99})
                elif _t[-1] == '初':
                    _time_list.append({'yn': 0})
                continue

            _year_match = _year_pattern.search(_t)
            if _year_match is not None:
                _g = _year_match.group()
                _time_list.append({'y': int(_g)})
                if _t.endswith('初'):
                    _time_list.append({'M': 1})
                elif _t.endswith('末') or _t.endswith('底'):
                    _time_list.append({'M': 12})
                continue

            _year_match_1 = _year_pattern_1.search(_t)
            if _year_match_1 is not None:
                _g = _year_match_1.group()
                _time_list.append({'y': int(_g)})
                continue

            _decade_match = _decade_pattern.search(_t)
            if _decade_match is not None:
                _g = _decade_match.group()
                if _t.endswith('初'):
                    _time_list.append({'dy': int(_g)})
                elif _t.endswith('末'):
                    _time_list.append({'dy': int(_g) + 9})
                elif _t.endswith('中期'):
                    _time_list.append({'dy': int(_g) + 5})
                elif _t.endswith('后期'):
                    _time_list.append({'dy': int(_g) + 7})
                else:
                    _time_list.append({'dy': int(_g)})
                continue

            _year_offset_match = _year_offset_pattern.search(_t)
            if _year_offset_match is not None:
                _g = _year_offset_match.group()
                if _t.startswith('今年') or _t.startswith('这1年') or _t.startswith('当年'):
                    _offset_list.update({'yy': 0})
                elif _t.startswith('明年') or _t.startswith('来年'):
                    _offset_list.update({'yy': 1})
                elif _t.startswith('后年'):
                    _offset_list.update({'yy': 2})
                elif _t.startswith('去年'):
                    _offset_list.update({'yy': -1})
                elif _t.startswith('前年') or _t.startswith('上年'):
                    _offset_list.update({'yy': -2})
                elif _t.startswith('大前年'):
                    _offset_list.update({'yy': -3})

                if _t[-1] == '初':
                    _time_list.append({'M': 1})
                elif _t[-1] == '末' or _t.endswith('底'):
                    _time_list.append({'M': 12})
                continue
                
            _year_offset_match1 = _year_offset_pattern1.search(_t)
            if _year_offset_match1 is not None:
                _g = _year_offset_match1.group()
                if '前' == _t[-1]:
                    _offset_list.update({'yy': -int(_g)})
                elif '后' == _t[-1]:
                    _offset_list.update({'yy': int(_g)})
                continue
                
            _season_match = _season_pattern.search(_t)
            if _season_match is not None:
                _g = _season_match.group()
                if _t.endswith('1季度'):
                    _time_list.append({'S': 1})
                elif _t.endswith('2季度'):
                    _time_list.append({'S': 2})
                elif _t.endswith('3季度'):
                    _time_list.append({'S': 3})
                elif _t.endswith('4季度'):
                    _time_list.append({'S': 4})
                continue

            _month_match = _month_pattern.search(_t)
            if _month_match is not None:
                _g = _month_match.group()
                _time_list.append({'M': int(_g)})
                if _t[-1] == '初':
                    _time_list.append({'d': 1})
                elif _t[-1] == '末' or _t[-1] == '底':
                    _time_list.append({'d': self.month_last_day(_now.year, int(_g))})
                continue

            _month_offset_match = _month_offset_pattern.search(_t)
            if _month_offset_match is not None:
                _g = _month_offset_match.group()
                if _t.startswith('上个月') or _t.startswith('上月'):
                    _offset_list.update({'yM': -1})
                elif _t.startswith('上上个月') or _t.startswith('上上月'):
                    _offset_list.update({'yM': -2})
                elif _t.startswith('下个月') or _t.startswith('下月') or _t.startswith('次月'):
                    _offset_list.update({'yM': 1})
                elif _t.startswith('下下个月') or _t.startswith('下下月'):
                    _offset_list.update({'yM': 2})
                elif _t.startswith('本月'):
                    _offset_list.update({'yM': 0})

                if _t.endswith('初'):
                    _time_list.append({'d': 1})
                elif _t.endswith('末') or _t.endswith('底'):
                    _time_list.append({'d': self.month_last_day(_now.year, _now.month)})
                continue

            _week_match = _week_pattern.search(_t)
            if _week_match is not None:
                _g = _week_match.group()
                if _t.startswith('上个星期') or _t.startswith('上个周') or _t.startswith('上个礼拜') or \
                        _t.startswith('上星期') or _t.startswith('上周') or _t.startswith('上礼拜'):
                    _offset_list.update({'yw': -1})
                elif _t.startswith('上上个星期') or _t.startswith('上上个周') or _t.startswith('上上个礼拜') or \
                        _t.startswith('上上星期') or _t.startswith('上上周') or _t.startswith('上上礼拜'):
                    _offset_list.update({'yw': -2})
                elif _t.startswith('下个星期') or _t.startswith('下个周') or _t.startswith('下个礼拜') or \
                        _t.startswith('下星期') or _t.startswith('下周') or _t.startswith('下礼拜'):
                    _offset_list.update({'yw': 1})
                elif _t.startswith('下下个星期') or _t.startswith('下下个周') or _t.startswith('下下个礼拜') or \
                        _t.startswith('下下星期') or _t.startswith('下下周') or _t.startswith('下下礼拜'):
                    _offset_list.update({'yw': 2})
                elif _t.startswith('这个星期') or _t.startswith('这个周') or _t.startswith('这个礼拜') or \
                        _t.startswith('这星期') or _t.startswith('这周') or _t.startswith('这礼拜'):
                    _time_list.append({'w': 0})
                if _t[-1] == '日' or _t[-1] == '天' or _t[-1] == '末':
                    _time_list.append({'w': 7})
                elif _t[-1] == '1' or _t[-1] == '2' or _t[-1] == '3' or _t[-1] == '4' or _t[-1] == '5' or _t[-1] == '6':
                    _time_list.append({'w': int(_t[-1])})
                continue

            _day_match = _day_pattern.search(_t)
            if _day_match is not None:
                _g = _day_match.group()
                _time_list.append({'d': int(_g)})
                continue

            _day_pattern_1_match = _day_pattern_1.search(_t)
            if _day_pattern_1_match is not None:
                _time_list.append(_day_pattern_1_mapping[_t])
                continue

            _day_offset_match = _day_offset_pattern.search(_t)
            if _day_offset_match is not None:
                _g = _day_offset_match.group()
                if _t == '今天' or _t == '今日':
                    _offset_list.update({'yd': 0})
                elif _t == '昨天' or _t == '昨日' or _t == '前1天':
                    _offset_list.update({'yd': -1})
                elif _t == '前天' or _t == '前日':
                    _offset_list.update({'yd': -2})
                elif _t == '大前天':
                    _offset_list.update({'yd': -3})
                elif _t == '明天' or _t == '明日':
                    _offset_list.update({'yd': 1})
                elif _t == '后天':
                    _offset_list.update({'yd': 2})
                elif _t == '大后天':
                    _offset_list.update({'yd': 3})
                continue

            _hour_match = _hour_pattern.search(_t)
            if _hour_match is not None:
                _g = _hour_match.group()
                _hour = int(_g)
                if _t.startswith('晚') and _hour < 12:
                    _time_list.append({'h': _hour + 12})
                elif _t.startswith('当晚') and _hour < 12:
                    _time_list.append({'h': _hour + 12})
                    _offset_list.update({'yd': 0})
                else:
                    _time_list.append({'h': int(_g)})
                if _t[-1] == '半':
                    _time_list.append({'m': 30})
                continue

            _hour_match_1 = _hour_pattern_1.search(_t)
            if _hour_match_1 is not None:
                if _t == '下午':
                    _pm_flag = True
                _time_list.append(_hour_pattern_1_mapping[_t])
                continue

            _minute_match = _minute_pattern.search(_t)
            if _minute_match is not None:
                _g = _minute_match.group()
                if _t.endswith('分钟'):
                    _ref = 'duration'
                    continue
                if _t[-1] == '后':
                    _offset_list.update({'ym': int(_g)})
                elif _t[-1] == '前':
                    _offset_list.update({'ym': -int(_g)})
                else:
                    _time_list.append({'m': int(_g)})
                continue

            _second_match = _second_pattern.search(_t)
            if _second_match is not None:
                _g = _second_match.group()
                if _t[-1] == '后':
                    _offset_list.update({'ys': int(_g)})
                elif _t[-1] == '前':
                    _offset_list.update({'ys': -int(_g)})
                else:
                    _time_list.append({'s': int(_g)})
                continue

            _dynasty_match = _dynasty_pattern.search(_t)
            if _dynasty_match is not None:
                _ref = 'dynasty'
                continue
            _duration_match = _duration_pattern.search(_t)
            if _duration_match is not None:
                _ref = 'duration'
                continue
        
        if _pm_flag:
            for _tl in _time_list:
                if 'h' in _tl.keys():
                    _offset_list.update({'yh': 12})
                    break
        self.time_unit.append({'text': ' '.join(ts['text']), 'time': _time_list,
                               'offset': _offset_list, 'start': ts['start'], 'end': ts['end'], 'ref': _ref})

    def find(self):
        self.match()
        self.resolve()

    def concat(self, end: int, start: int):
        """
        判断两个时间表达式是否应该链接成一个
        :param end: 上一个结束位置
        :param start: 下一个开始位置
        :return: 是否连接，
        """
        if end == -1:
            return False
        if self.regular_sequence[end - 1] in ['至', '和']:
            return False
        if start == end:
            return True
        elif abs(start - end) > 3:
            return False
        else:
            _sub_string = self.regular_sequence[end: start]
            if len(_sub_string) == 1 and _sub_string in {'前', '后', '头'}:
                return True
            if '。' in _sub_string:
                return False
            _join_list = {'(', ')', '（', '）'}
            for _c in _join_list:
                if _c in _sub_string:
                    return True
        return False

    def find_context_time(self, _i):
        """
        找到合适的上下文时间
        :return:
        """
        if _i == 0:
            return self.base_time
        else:
            _distance = abs(self.time_unit[_i]['start'] - self.time_unit[_i - 1]['end'])
            if _distance < 3:
                _current_offset = self.time_unit[_i]['offset']
                for _k, _v in self.time_unit[_i - 1]['offset'].items():
                    if _k not in _current_offset.keys():
                        _current_offset[_k] = _v
                if 'final_complete' in self.time_unit[_i - 1]:
                    return arrow.get(self.time_unit[_i - 1]['final_complete'])
                else:
                    return self.base_time
            elif _distance <= 10:
                if 'final_complete' in self.time_unit[_i - 1]:
                    return arrow.get(self.time_unit[_i - 1]['final_complete'])
                else:
                    return self.base_time
            else:
                return self.base_time
                
    @staticmethod
    def prefilter(sequence):
        sequence = StringUtil.number_translate(sequence)
        nums = {'０': '0', '１': '1', '２': '2', '３': '3', '４': '4', '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'}
        for k, v in nums.items():
            sequence = sequence.replace(k, v)
        return sequence

    def is_time(self, _tu):
        _scale = 10
        _sub_string = self.regular_sequence[max(_tu['start'] - _scale, 0): min(len(self.regular_sequence), _tu['end'] + _scale)]
        _tu['context'] = _sub_string
        # return 1
        _seq = tokenizer.texts_to_sequences([_sub_string])
        _data = keras.preprocessing.sequence.pad_sequences(_seq, maxlen=25)
        _pred = model.predict(_data.reshape((1, -1)))
        _tu['pred'] = str(_pred[0][0])
        return _pred[0][0]
        


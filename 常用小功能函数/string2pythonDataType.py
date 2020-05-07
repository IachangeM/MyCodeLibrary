"""
  虽然我也不知道当时为啥写这个了，但是先留着吧！
"""

# 将这些特殊字符串(yes/no/True/False/on/off)解析成布尔值
BOOLEAN_STATES = {'yes': True, 'true': True, 'on': True,
                  'no': False, 'false': False, 'off': False}
                  
def _convert_to_boolean(value):
    """Return a boolean value translating from other types if necessary.
    """
    if value.lower() not in BOOLEAN_STATES:
        raise ValueError('Not a boolean: %s' % value)
    return BOOLEAN_STATES[value.lower()]


def parse_value(value):
    """在使用_convert_to_boolean/int/float/str将字符串解析成python对应数据类型的时候
    注意：
        整数可以被解析成小数，所以一定先尝试解析成整数
        bool值可以被解析成字符串
    因此尝试解析的顺序为：
        boolean > int > float > str
    :param value:字符串类型的参数值
    :return:对应python数据类型的参数值
    """
    order_functions = [_convert_to_boolean, int, float, str]
    for fun in order_functions:
        try:
            return fun(value)
        except ValueError:
            pass  # continue is also ok.


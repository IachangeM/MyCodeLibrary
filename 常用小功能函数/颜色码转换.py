


def htmlcolor2rgb(color_code: str) -> tuple:
    """
    Args:
        color_code: html格式的颜色码：#22AAFF
    Returns: tuple(r, g, b)
    """
    assert type(color_code) == str
    assert len(color_code) == 7
    assert color_code[0] == '#'

    legal_values = [str(i) for i in range(11)] + ['A', 'B', 'C', 'D', 'E', 'F']

    # 1. 转换成大写, 并检查
    color_code = color_code.upper()
    for c in color_code[1:]:
        if c not in legal_values:
            raise ValueError('Unlegal char `{}` in html color code.'.format(c))

    return tuple(
        map(lambda p: int(p, base=16), [color_code[1:3], color_code[3:5], color_code[5:7]])
    )





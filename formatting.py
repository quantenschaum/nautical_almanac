from decimal import Decimal, ROUND_HALF_UP
from math import floor, copysign

_markers = {}
_decimals = 1
_round = round


def decimals(n):
    global _decimals
    _decimals = n


def marker(k, m):
    _markers[k] = m


def replace(s, m={}):
    for k, v in m.items():
        s = s.replace(k, v)
    for k in _markers.keys():
        s = s.replace(k, m.get(k) or _markers[k])
    return s


def round(x, n=0, mode=ROUND_HALF_UP):
    "https://realpython.com/python-rounding/"
    if mode is None:
        x = _round(x, n)
        return int(x) if n == 0 or x == 0 else x
    assert x == float(str(x))
    x = Decimal(str(x)).quantize(Decimal("1." + "0" * n), mode)
    return int(x) if n == 0 or x == 0 else float(x)


def deg_min(a, n=None):
    "decimal degrees -> integer degrees, decimal minutes with n digits"
    n = _decimals if n is None else n
    b = abs(a)
    d, m = floor(b), round(60 * (b % 1), n)
    if m >= 60:
        d += 1
        m -= 60
    return copysign(d % 360, a), m


def is_number(v):
    return isinstance(v, int) or isinstance(v, float)


def f(v, s=""):
    "format v with format s for numbers as +00.0"
    if is_number(v):
        if isinstance(s, int):
            w = f(v, " 0." + "0" * s) if s else f(v, " 0")
        elif isinstance(s, str):
            if "{" in s:
                w = s.format(v)
            else:
                s = s if "0" in s else s + " 0.0"  # default format if nothing or only sign is given
                m = len(s)  # total width of string
                n = s.split(".")[1].count("0") if "." in s else 0  # digits after .
                p = "+" if "+" in s else " " if "-" in s else "-"  # sign prefix
                p += "0" if s.split(".")[0].count("0") > 1 else ""  # zero padding
                u = s[s.rfind("0") + 1:]  # suffix (unit)
                m -= len(u)  # suffix does not count into total width
                w = f(round(v, n), f"{{:{p}{m}.{n}f}}{u}")
    else:
        w = s.format(v) if isinstance(s, str) and "{" in s else f"{v:{s}}"

    return replace(w) if is_number(v) else w


def dm(a, n=None):
    "format as degrees and minutes: 000°00.0'"
    n = _decimals if n is None else n
    d, m = deg_min(a, n)
    k = (n + 3 if n else 2)
    return replace(f"{d:3.0f}°{m:{k}.{n}f}'")


def hms(H, p=""):
    "format as HH:MM:SS"
    if H is None:
        return "--:--:--"
    h = int(abs(H))
    m = int(60 * (abs(H) % 1))
    s = round(60 * (abs(60 * H) % 1))
    if s >= 60:
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        h += 1
    p = p.replace("-", " ")
    n = 3 if p or H < 0 else 2
    h = copysign(h, H)
    return replace(f"{h:{p}0{n}.0f}:{m:02.0f}:{s:02.0f}")


def hm(H, s=""):
    "format hours as HH:MM"
    if H is None:
        return "--:--"
    h = int(abs(H))
    m = round(60 * (abs(H) % 1))
    if m >= 60:
        m -= 60
        h += 1
    s = s.replace("-", " ")
    n = 3 if s or H < 0 else 2
    h = copysign(h, H)
    return replace(f"{h:{s}0{n}.0f}:{m:02.0f}")


def angle(s):
    "parse 'deg min sec'"
    for c in "°':":
        s = s.replace(c, " ")
    dms = s.split()
    assert 0 < len(dms) < 4
    dms = [float(v) for v in dms]
    a = sum([abs(v) / pow(60, i) for i, v in enumerate(dms)])
    return copysign(a, dms[0])


def parse(s):
    "parse int, float, angle or leave as is"
    for t in int, float, angle:
        try:
            return t(s)
        except:
            pass
    return s

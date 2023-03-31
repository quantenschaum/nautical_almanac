#!/usr/bin/env python
import atexit
import pickle
from collections import defaultdict
from datetime import datetime, date
from datetime import timedelta as duration
from decimal import Decimal, ROUND_HALF_UP
from math import copysign, floor, cos, nan, sqrt, atan, sin, asin, acos, radians, degrees
from os.path import isfile

from cachetools import cached
from numpy import array, arange
from progress.bar import Bar
from skyfield import almanac
from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos, iers
from skyfield.earthlib import refraction
from skyfield.framelib import itrs
from skyfield.magnitudelib import planetary_magnitude
from skyfield.timelib import Time

try:
    from matplotlib import use
    from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, gca, xlim, ylim, savefig, tight_layout, legend
    import seaborn as sns
except:
    raise

planets = "Venus", "Mars", "Jupiter", "Saturn"

stars = {"Alpheratz": 677, "Ankaa": 2081, "Schedar": 3179, "Diphda": 3419, "Achernar": 7588, "Hamal": 9884,
         "Acamar": 13847, "Menkar": 14135, "Mirfak": 15863, "Aldebaran": 21421, "Rigel": 24436, "Capella": 24608,
         "Bellatrix": 25336, "Elnath": 25428, "Alnilam": 26311, "Betelgeuse": 27989, "Canopus": 30438, "Sirius": 32349,
         "Adhara": 33579, "Procyon": 37279, "Pollux": 37826, "Avior": 41037, "Suhail": 44816, "Miaplacidus": 45238,
         "Alphard": 46390, "Regulus": 49669, "Dubhe": 54061, "Denebola": 57632, "Gienah": 59803, "Acrux": 60718,
         "Gacrux": 61084, "Alioth": 62956, "Spica": 65474, "Alkaid": 67301, "Hadar": 68702, "Menkent": 68933,
         "Arcturus": 69673, "Rigil Kent.": 71683, "Zuben'ubi": 72622, "Kochab": 72607, "Alphecca": 76267,
         "Antares": 80763, "Atria": 82273, "Sabik": 84012, "Shaula": 85927, "Rasalhague": 86032, "Eltanin": 87833,
         "Kaus Aust.": 90185, "Vega": 91262, "Nunki": 92855, "Altair": 97649, "Peacock": 100751, "Deneb": 102098,
         "Enif": 107315, "Alnair": 109268, "Fomalhaut": 113368, "Markab": 113963, }

latitudes = (72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 45, 40, 35, 30, 20, 10, 0,
             -10, -20, -30, -35, -40, -45, -50, -52, -54, -56, -58, -60)

_caches = defaultdict(dict)


def load_cache(filename="cache.pkl"):
    with open(filename, "rb") as f:
        for k, c in pickle.load(f).items():
            _caches[k].update(c)


def dump_cache(filename="cache.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(_caches, f)


def iers_dates(finals):
    "return dates when Polar motion, Time, Nutation data (I) and predictions (P) ends"
    dates = defaultdict(dict)
    dt0 = None
    flags0 = None
    keys = "PTN"  # polar motion, time, nutation
    with open(finals) as file:
        for line in file:
            mjd = int(line[7:12])
            if mjd < 51544:
                continue  # skip data before 2000

            y, m, d = [int(line[i:i + 2]) for i in (0, 2, 4)]
            dt = date(y + 2000, m, d)
            flags = [line[i:i + 1] for i in (16, 57, 95)]

            if flags0:
                for i, k in enumerate(keys):
                    f, f0 = flags[i], flags0[i]
                    if f != f0:
                        dates[k][f0] = dt0

            dt0 = dt
            flags0 = flags

    return dates


_iers_info = None


def iers_info(kind, mode):
    return _iers_info[kind][mode] if _iers_info else None


ARIES = "Aries"


def init(iers_time=True, polar_motion=True, ephemeris="de440s", cache=False):
    "initialize SkyField and Almanac global state"
    global ts, eph, bodies, earth, star_df, _iers_info

    finals = "finals2000A.all"

    if iers_time:
        if not isfile(finals):
            load.download("https://datacenter.iers.org/data/9/" + finals)
        _iers_info = iers_dates(finals)

    ts = load.timescale(builtin=not iers_time)

    if iers_time and polar_motion:
        with open(finals) as f:
            data = iers.parse_x_y_dut1_from_finals_all(f)
            iers.install_polar_motion_table(ts, data)

    ephemeris += "" if ephemeris.endswith(".bsp") else ".bsp"
    if not isfile(ephemeris):
        load.download(ephemeris)

    eph = load(ephemeris)

    hipdat = "hip_main.dat"

    if not isfile(hipdat):
        load.download(hipparcos.url)

    with open(hipdat) as f:
        star_df = hipparcos.load_dataframe(f)

    bodies = {k: Star.from_dataframe(star_df.loc[v]) for (k, v) in stars.items()}

    earth = eph["earth"]
    bodies.update({
        "Earth": earth,
        "Sun": eph["sun"],
        "Moon": eph["moon"],
        "Venus": eph["venus barycenter"],
        "Mars": eph["mars barycenter"],
        "Jupiter": eph["jupiter barycenter"],
        "Saturn": eph["saturn barycenter"],
        "Mercury": eph["mercury barycenter"],
        ARIES: Star(ra_hours=0, dec_degrees=0),
        "Polaris": Star.from_dataframe(star_df.loc[11767]),
    })

    if cache:
        try:
            load_cache()
        except:
            pass

        atexit.register(dump_cache)


def body(name):
    "return ephemeris for named body"
    return bodies[name]


def time(t):
    "python datetime (UT1) -> skyfield time"
    if isinstance(t, Time):
        return t
    # return ts.from_datetime(t.replace(tzinfo=utc))
    return ts.ut1(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)


def dtime(t):
    "skyfield time -> python datetime (UT1, rounded to integer seconds)"
    # return t.utc_datetime().replace(tzinfo=None)
    c = list(t.ut1_calendar())
    c[5] = round(c[5])
    if c[5] >= 60:
        c[5] -= 60
        return datetime(*c) + duration(minutes=1)
    return datetime(*c)


def hours(t):
    "time -> decimal hours of day (since 00:00)"
    t = dtime(t)
    return t.hour + (t.minute + (t.second + t.microsecond / 1e6) / 60) / 60


def delta_ut1(t):
    "difference UT1-UTC in s (should be <0.9s)"
    return float(time(t).dut1)


def delta_t(t):
    "difference TT-UT1 in s"
    return float(time(t).delta_t)


_gha = 0  # 0=radec+gast, 1=latlon(itrs), 2=like 1 w/o M, 3=like 1, only Aries w/o M


# @cached(_caches["sha"])
def sha_dec(t, b):
    "SHA and Dec of body b at time t in degrees"
    if _gha == 0:
        if b == ARIES:
            return 0.0, 0.0
        t = time(t)
        ra, dec, _ = earth.at(t).observe(bodies[b]).apparent().radec("date")
        return -ra._degrees % 360, dec.degrees
    gha, dec = gha_dec(t, b)
    sha = (gha - gha_dec(t, ARIES)[0]) % 360
    return sha, dec


# @cached(_caches["gha"])
def gha_dec(t, b):
    "GHA and Dec of body b at time t in degrees"
    if _gha == 0 or _gha == 2 and b == ARIES:
        if b == ARIES:
            return 15 * time(t).gast, 0.0  # ICRS right ascension (0) + gast
        sha, dec = sha_dec(t, b)
        return (sha + gha_dec(t, ARIES)[0]) % 360, dec

    # using frame_latlon instead of radec+gast also aplies time.M and polar_motion_matrix
    # which results in ITRS GHA, Dec but do not match the values from the almanac
    t = time(t)
    lat, lon, _ = earth.at(t).observe(bodies[b]).apparent().frame_latlon(itrs)
    dec, gha = lat.degrees, -lon.degrees % 360
    return gha, dec


def is_number(v):
    return isinstance(v, int) or isinstance(v, float)


def gha_comparison():
    global _gha
    init()
    t = datetime(2023, 1, 1, 13)
    print(t, "UT1")
    print("Object      M   GHA          Dec          SHA          Dec")
    for b in ARIES, "Sun", "Vega":
        for i in range(3):
            _gha = i
            a = [b, i]
            a += gha_dec(t, b)
            a += sha_dec(t, b)
            s = ""
            for k, v in enumerate(a):
                s += f"{v:12.8f} " if isinstance(v, float) else f"{v:6} "
            print(s)


def lha_dec(t, b, lon):
    "LHA and Dec of body b at time t in degrees"
    gha, dec = gha_dec(t, b)
    return (gha + lon) % 360, dec


def alt_az(t, b, lat, lon, sky=0):
    "Altitude and Azimuth of body b at time t at (lat,lon) in degrees"
    t = time(t)
    if sky:
        # this includes parallax at pos on earth surface and dip and optionally refraction
        l = earth + wgs84.latlon(lat, lon)
        alt, az, dist = l.at(t).observe(bodies[b]).apparent().altaz()
        return alt.degrees, az.degrees
    lha, dec = lha_dec(t, b, lon)
    lha, dec, lat = radians(lha), radians(dec), radians(lat)
    clha = cos(lha)
    sdec, cdec = sin(dec), cos(dec)
    slat, clat = sin(lat), cos(lat)
    hc = asin(sdec * slat + clat * cdec * clha)
    shc, chc = sin(hc), cos(hc)
    z = acos((sdec - slat * shc) / (clat * chc))
    hc, z = degrees(hc), degrees(z)
    zn = z if lha > 180 else 360 - z
    return hc, zn


@cached(_caches["sd"])
def semi_diameter(t, b):
    "semi diameter of body b at time t in arc minutes"
    t = time(t)
    _, _, dist = earth.at(t).observe(bodies[b]).apparent().radec(t)
    radius = {"Sun": 695997, "Moon": 1739.9}  # km
    return degrees(atan(radius[b] / dist.km)) * 60


def hp_moon(t):
    "moon's horizontal parallax at time t in arc minutes"
    return semi_diameter(t, "Moon") / 0.272805950305


def d_value(t, b):
    "d value of body b (rate of change of DEC) in arc minutes/hour"
    gha0, dec0 = gha_dec(t, b)
    gha1, dec1 = gha_dec(t + duration(hours=1), b)
    dec0, dec1 = dec0 * 60, dec1 * 60
    # dec0, dec1 = round(dec0, 1) / 60, round(dec1, 1) / 60
    return abs(dec1) - abs(dec0) if dec0 * dec1 > 0 else (dec1 - dec0)


def v_value(t, b):
    "v values of body b (excess rate of change of GHA) in arc minutes/hour"
    gha0, dec0 = gha_dec(t, b)
    gha1, dec1 = gha_dec(t + duration(hours=1), b)
    base = (14 + 19 / 60) if b == "Moon" else 15
    # gha0, gha1 = round(gha0 * 60, 1) / 60, round(gha1 * 60, 1) / 60
    return ((gha1 - gha0) % 360 - base) * 60


@cached(_caches["mag"])
def magnitude(t, b):
    "magnitude of body b at time t"
    t = time(t)
    if b in stars:
        return star_df.loc[stars[b]].magnitude
    else:
        m = planetary_magnitude(earth.at(t).observe(bodies[b]).apparent())
        return float(m)


def equation_of_time(t):
    "equation of time (solar time - UT1) at time t in minutes"
    gha, dec = gha_dec(t, "Sun")
    tsun = (gha / 15 - 12) % 24  # solar time
    tut1 = hours(time(t))
    eqot = tsun - tut1
    if abs(eqot) > 1:
        eqot -= copysign(24, eqot)
    return eqot * 60


@cached(_caches["mp"])
def meridian_passage(t, b, lon=0, upper=True):
    "time of meridian passage of body b at date t in hours"
    f = almanac.meridian_transits(eph, bodies[b], wgs84.latlon(0, lon))
    t0, t1 = time(t), time(t + duration(hours=24))
    times, events = almanac.find_discrete(t0, t1, f)
    times = times[events == bool(upper)]
    #     assert len(times) <= 1,(s,t0,t1,times)
    if not times:
        return None
    return hours(times[0])


@cached(_caches["sr"])
def sunrise_sunset(t, lat, lon=0):
    "sunset/sunrise times 1=rise 0=set"
    t0, t1 = time(t), time(t + duration(hours=24))
    f = almanac.sunrise_sunset(eph, wgs84.latlon(lat, lon))
    times, events = almanac.find_discrete(t0, t1, f)
    f0 = int(f(t0))
    data = {"t0": f0, "min": f0, "max": f0}
    for t, e in zip(times, events):
        data[e] = hours(t)
        data["min"] = min(data["min"], e)
        data["max"] = max(data["max"], e)
    return data


@cached(_caches["tw"])
def twilight(t, lat, lon=0):
    """twilights and sunset/sunrise
    0 = night, dark
    1 = astronomical twilight
    2 = nautical twilight
    3 = civil twilight
    4 = day, sun is up
    negative sign, if sun is going down
    """
    t0, t1 = time(t), time(t + duration(hours=24))
    f = almanac.dark_twilight_day(eph, wgs84.latlon(lat, lon))
    f.step_days = 0.01
    times, events = almanac.find_discrete(t0, t1, f)
    f0 = int(f(t0))
    data = {"t0": f0, "min": f0, "max": f0}
    for t, e in zip(times, events):
        w = e if e > f0 else -e
        # assert w not in data, (dtime(t), w, data, times, events)
        if w not in data:
            data[w] = hours(t)
        data["min"] = min(data["min"], e)
        data["max"] = max(data["max"], e)
        f0 = e
    return data


@cached(_caches["mr"])
def moon_rise_set(t, lat, lon=0):
    "time of moon rise and set"
    s = "Moon"
    sd = semi_diameter(t, s)
    f = almanac.risings_and_settings(eph, bodies[s], wgs84.latlon(lat, lon), radius_degrees=sd)
    f.step_days = 0.01
    t0, t1 = time(t), time(t + duration(hours=24))
    times, events = almanac.find_discrete(t0, t1, f)
    f0 = int(f(t0))
    data = {"t0": f0, "min": f0, "max": f0}
    for t, e in zip(times, events):
        data[e if e not in data else e + 10] = hours(t)
        data["min"] = min(data["min"], e)
        data["max"] = max(data["max"], e)
    return data


@cached(_caches["ma"])
def moon_age_phase(t):
    "moon age (time since new moon) in days and phase (fraction illuminated)"
    t0, t1 = time(t - duration(days=30)), time(t + duration(hours=24))
    p = almanac.moon_phase(eph, time(t)).radians
    illum = (1 - cos(p)) / 2
    times, events = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
    new_moon = times[events == 0][0]
    age = t - dtime(new_moon)
    age = age.total_seconds() / 3600 / 24
    return age, illum


def inc_sun(m, s):
    "GHA increment of sun"
    h = (m + s / 60) / 60
    return h * 15


def inc_aries(m, s):
    "GHA increment of Aries"
    h = (m + s / 60) / 60
    return h * (15 + 2.46 / 60)


def inc_moon(m, s):
    "GHA increment of moon"
    h = (m + s / 60) / 60
    return h * (14 + 19 / 60)


def v_corr(m, s, v):
    "v/d correction"
    h = (m + s / 60) / 60
    return v * h


_markers = {}
_decimals = 1


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


round0 = round

_rounding = ROUND_HALF_UP


def round(x, n=0):
    "https://realpython.com/python-rounding/"
    assert x == float(str(x))
    x = Decimal(str(x)).quantize(Decimal("1." + n * "0"), _rounding)
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


def f(v, s=None):
    if is_number(v):
        if s is None:
            w = f(v, 1)
        elif isinstance(s, int):
            w = f(v, "00." + "0" * s) if s else f(v, "00")
        elif isinstance(s, str):
            if "{" in s:
                w = s.format(v)
            else:
                s = s if "0" in s else s + "00.0"
                m = len(s)
                n = s.split(".")[1].count("0") if "." in s else 0
                p = "+" if "+" in s else "-"
                u = s[s.rfind("0") + 1:]
                m -= len(u)
                w = f(round(v, n), f"{{:{p}{m}.{n}f}}{u}")
    elif s is None:
        w = str(v)
    else:
        w = s.format(v) if "{" in s else f"{{:{s}}}".format(v)

    return replace(w) if is_number(v) else w


def dm(a, n=None):
    "format as degrees and minutes: 000°00.0'"
    n = _decimals if n is None else n
    d, m = deg_min(a, n)
    k = (n + 3 if n else 2)
    return replace(f"{d:3.0f}°{m:{k}.{n}f}'")


def hms(H, s=False, rep={}):
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
    pad = "-" if H < 0 else " " if s else ""
    return replace(f"{pad}{h:02.0f}:{m:02.0f}:{s:02.0f}", rep)


def hm(H, s=False, rep={}):
    "format hours as HH:MM"
    if H is None:
        return "--:--"
    h = int(abs(H))
    m = round(60 * (abs(H) % 1))
    if m >= 60:
        m -= 60
        h += 1
    pad = "-" if H < 0 else " " if s else ""
    return replace(f"{pad}{h:02.0f}:{m:02.0f}", rep)


def ms(H, s=False, rep={}):
    "format minutes as MM:SS"
    if H is None:
        return "--:--"
    m = int(abs(H))
    s = round(60 * (abs(H) % 1))
    if s >= 60:
        s -= 60
        m += 1
    pad = "-" if H < 0 else " " if s else ""
    return replace(f"{pad}{m:02.0f}:{s:02.0f}", rep)


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


_alt_corr10 = (
    # alitude sun winter  sun summer  planets
    # deg min lower upper lower upper stars
    (0, 0, -17.5, -49.8, -17.8, -49.6, -33.8),
    (0, 3, -16.9, -49.2, -17.2, -49.0, -33.2),
    (0, 6, -16.3, -48.6, -16.6, -48.4, -32.6),
    (0, 9, -15.7, -48.0, -16.0, -47.8, -32.0),
    (0, 12, -15.2, -47.5, -15.5, -47.3, -31.5),
    (0, 15, -14.6, -46.9, -14.9, -46.7, -30.9),
    (0, 18, -14.1, -46.4, -14.4, -46.2, -30.4),
    (0, 21, -13.5, -45.8, -13.8, -45.6, -29.8),
    (0, 24, -13.0, -45.3, -13.3, -45.1, -29.3),
    (0, 27, -12.5, -44.8, -12.8, -44.6, -28.8),
    (0, 30, -12.0, -44.3, -12.3, -44.1, -28.3),
    (0, 33, -11.6, -43.9, -11.9, -43.7, -27.9),
    (0, 36, -11.1, -43.4, -11.4, -43.2, -27.4),
    (0, 39, -10.6, -42.9, -10.9, -42.7, -26.9),
    (0, 42, -10.2, -42.5, -10.5, -42.3, -26.5),
    (0, 45, -9.8, -42.1, -10.1, -41.9, -26.1),
    (0, 48, -9.4, -41.7, -9.7, -41.5, -25.7),
    (0, 51, -9.0, -41.3, -9.3, -41.1, -25.3),
    (0, 54, -8.6, -40.9, -8.9, -40.7, -24.9),
    (0, 57, -8.2, -40.5, -8.5, -40.3, -24.5),
    (1, 0, -7.8, -40.1, -8.1, -39.9, -24.1),
    (1, 3, -7.4, -39.7, -7.7, -39.5, -23.7),
    (1, 6, -7.1, -39.4, -7.4, -39.2, -23.4),
    (1, 9, -6.7, -39.0, -7.0, -38.8, -23.0),
    (1, 12, -6.4, -38.7, -6.7, -38.5, -22.7),
    (1, 15, -6.0, -38.3, -6.3, -38.1, -22.3),
    (1, 18, -5.7, -38.0, -6.0, -37.8, -22.0),
    (1, 21, -5.4, -37.7, -5.7, -37.5, -21.7),
    (1, 24, -5.1, -37.4, -5.4, -37.2, -21.4),
    (1, 27, -4.8, -37.1, -5.1, -36.9, -21.1),
    (1, 30, -4.5, -36.8, -4.8, -36.6, -20.8),
    (1, 35, -4.0, -36.3, -4.3, -36.1, -20.3),
    (1, 40, -3.6, -35.9, -3.9, -35.7, -19.9),
    (1, 45, -3.1, -35.4, -3.4, -35.2, -19.4),
    (1, 50, -2.7, -35.0, -3.0, -34.8, -19.0),
    (1, 55, -2.3, -34.6, -2.6, -34.4, -18.6),
    (2, 0, -1.9, -34.2, -2.2, -34.0, -18.2),
    (2, 5, -1.5, -33.8, -1.8, -33.6, -17.8),
    (2, 10, -1.1, -33.4, -1.4, -33.2, -17.4),
    (2, 15, -0.8, -33.1, -1.1, -32.9, -17.1),
    (2, 20, -0.4, -32.7, -0.7, -32.5, -16.7),
    (2, 25, -0.1, -32.4, -0.4, -32.2, -16.4),
    (2, 30, 0.2, -32.1, -0.1, -31.9, -16.1),
    (2, 35, 0.5, -31.8, 0.2, -31.6, -15.8),
    (2, 40, 0.9, -31.4, 0.6, -31.2, -15.4),
    (2, 45, 1.1, -31.2, 0.8, -31.0, -15.2),
    (2, 50, 1.4, -30.9, 1.1, -30.7, -14.9),
    (2, 55, 1.7, -30.6, 1.4, -30.4, -14.6),
    (3, 0, 2.0, -30.3, 1.7, -30.1, -14.3),
    (3, 5, 2.2, -30.1, 1.9, -29.9, -14.1),
    (3, 10, 2.5, -29.8, 2.2, -29.6, -13.8),
    (3, 15, 2.7, -29.6, 2.4, -29.4, -13.6),
    (3, 20, 2.9, -29.4, 2.6, -29.2, -13.4),
    (3, 25, 3.2, -29.1, 2.9, -28.9, -13.1),
    (3, 30, 3.4, -28.9, 3.1, -28.7, -12.9),
    (3, 35, 3.6, -28.7, 3.3, -28.5, -12.7),
    (3, 40, 3.8, -28.5, 3.5, -28.3, -12.5),
    (3, 45, 4.0, -27.9, 3.8, -28.0, -12.3),
    (3, 50, 4.2, -28.1, 3.9, -27.9, -12.1),
    (3, 55, 4.4, -27.9, 4.1, -27.7, -11.9),
    (4, 0, 4.6, -27.7, 4.3, -27.5, -11.7),
    (4, 5, 4.8, -27.5, 4.5, -27.3, -11.5),
    (4, 10, 4.9, -27.4, 4.6, -27.2, -11.4),
    (4, 15, 5.1, -27.2, 4.8, -27.0, -11.2),
    (4, 20, 5.3, -27.0, 5.0, -26.8, -11.0),
    (4, 25, 5.4, -26.9, 5.1, -26.7, -10.9),
    (4, 30, 5.6, -26.7, 5.3, -26.5, -10.7),
    (4, 35, 5.7, -26.6, 5.4, -26.4, -10.6),
    (4, 40, 5.9, -26.4, 5.6, -26.2, -10.4),
    (4, 45, 6.0, -26.3, 5.7, -26.1, -10.3),
    (4, 50, 6.2, -26.1, 5.9, -25.9, -10.1),
    (4, 55, 6.3, -26.0, 6.0, -25.8, -10.0),
    (5, 0, 6.5, -25.8, 6.2, -25.6, -9.8),
    (5, 5, 6.6, -25.7, 6.3, -25.5, -9.7),
    (5, 10, 6.7, -25.6, 6.4, -25.4, -9.6),
    (5, 15, 6.8, -25.5, 6.5, -25.3, -9.5),
    (5, 20, 7.0, -25.3, 6.7, -25.1, -9.3),
    (5, 25, 7.1, -25.2, 6.8, -25.0, -9.2),
    (5, 30, 7.2, -25.1, 6.9, -24.9, -9.1),
    (5, 35, 7.3, -25.0, 7.0, -24.8, -9.0),
    (5, 40, 7.4, -24.9, 7.1, -24.7, -8.9),
    (5, 45, 7.5, -24.8, 7.2, -24.6, -8.8),
    (5, 50, 7.6, -24.7, 7.3, -24.5, -8.7),
    (5, 55, 7.7, -24.6, 7.4, -24.4, -8.6),
    (6, 0, 7.8, -24.5, 7.5, -24.3, -8.5),
    (6, 10, 8.0, -24.3, 7.7, -24.1, -8.3),
    (6, 20, 8.2, -24.1, 7.9, -23.9, -8.1),
    (6, 30, 8.4, -23.9, 8.1, -23.7, -7.9),
    (6, 40, 8.6, -23.7, 8.3, -23.5, -7.7),
    (6, 50, 8.7, -23.6, 8.4, -23.4, -7.6),
    (7, 0, 8.9, -23.4, 8.6, -23.2, -7.4),
    (7, 10, 9.1, -23.2, 8.8, -23.0, -7.2),
    (7, 20, 9.2, -23.1, 8.9, -22.9, -7.1),
    (7, 30, 9.4, -22.9, 9.1, -22.7, -6.9),
    (7, 40, 9.5, -22.8, 9.2, -22.6, -6.8),
    (7, 50, 9.6, -22.7, 9.3, -22.5, -6.7),
    (8, 0, 9.7, -22.6, 9.4, -22.4, -6.6),
    (8, 10, 9.9, -22.4, 9.6, -22.2, -6.4),
    (8, 20, 10.0, -22.3, 9.7, -22.1, -6.3),
    (8, 30, 10.1, -22.2, 9.8, -22.0, -6.2),
    (8, 40, 10.2, -22.1, 9.9, -21.9, -6.1),
    (8, 50, 10.3, -22.0, 10.0, -21.8, -6.0),
    (9, 0, 10.4, -21.9, 10.1, -21.7, -5.9),
    (9, 10, 10.5, -21.8, 10.2, -21.6, -5.8),
    (9, 20, 10.6, -21.7, 10.3, -21.5, -5.7),
    (9, 30, 10.7, -21.6, 10.4, -21.4, -5.6),
    (9, 40, 10.8, -21.5, 10.5, -21.3, -5.5),
    (9, 50, 10.9, -21.4, 10.6, -21.2, -5.4),
    (10, 0, 11.0, -21.3, 10.7, -21.1, -5.3),
)

_alt_corr90 = (
    # sun   winter      sun summer           stars
    # d, m,  L,    U,   d, m,   L,     U,   d,  m,  c
    (9, 33, 10.8, -21.5, 9, 39, 10.6, -21.2, 9, 55, -5.3),
    (9, 45, 10.9, -21.4, 9, 50, 10.7, -21.1, 10, 7, -5.2),
    (9, 56, 11.0, -21.3, 10, 2, 10.8, -21.0, 10, 20, -5.1),
    (10, 8, 11.1, -21.2, 10, 14, 10.9, -20.9, 10, 32, -5.0),
    (10, 20, 11.2, -21.1, 10, 27, 11.0, -20.8, 10, 46, -4.9),
    (10, 33, 11.3, -21.0, 10, 40, 11.1, -20.7, 10, 59, -4.8),
    (10, 46, 11.4, -20.9, 10, 53, 11.2, -20.6, 11, 14, -4.7),
    (11, 0, 11.5, -20.8, 11, 7, 11.3, -20.5, 11, 29, -4.6),
    (11, 15, 11.6, -20.7, 11, 22, 11.4, -20.4, 11, 44, -4.5),
    (11, 30, 11.7, -20.6, 11, 37, 11.5, -20.3, 12, 0, -4.4),
    (11, 45, 11.8, -20.5, 11, 53, 11.6, -20.2, 12, 17, -4.3),
    (12, 1, 11.9, -20.4, 12, 10, 11.7, -20.1, 12, 35, -4.2),
    (12, 18, 12.0, -20.3, 12, 27, 11.8, -20.0, 12, 53, -4.1),
    (12, 36, 12.1, -20.2, 12, 45, 11.9, -19.9, 13, 12, -4.0),
    (12, 54, 12.2, -20.1, 13, 4, 12.0, -19.8, 13, 32, -3.9),
    (13, 14, 12.3, -20.0, 13, 24, 12.1, -19.7, 13, 53, -3.8),
    (13, 34, 12.4, -19.9, 13, 44, 12.2, -19.6, 14, 16, -3.7),
    (13, 55, 12.5, -19.8, 14, 6, 12.3, -19.5, 14, 39, -3.6),
    (14, 17, 12.6, -19.7, 14, 29, 12.4, -19.4, 15, 3, -3.5),
    (14, 41, 12.7, -19.6, 14, 53, 12.5, -19.3, 15, 29, -3.4),
    (15, 5, 12.8, -19.5, 15, 18, 12.6, -19.2, 15, 56, -3.3),
    (15, 31, 12.9, -19.4, 15, 45, 12.7, -19.1, 16, 25, -3.2),
    (15, 59, 13.0, -19.3, 16, 13, 12.8, -19.0, 16, 55, -3.1),
    (16, 27, 13.1, -19.2, 16, 43, 12.9, -18.9, 17, 27, -3.0),
    (16, 58, 13.2, -19.1, 17, 14, 13.0, -18.8, 18, 1, -2.9),
    (17, 30, 13.3, -19.0, 17, 47, 13.1, -18.7, 18, 37, -2.8),
    (18, 5, 13.4, -18.9, 18, 23, 13.2, -18.6, 19, 16, -2.7),
    (18, 41, 13.5, -18.8, 19, 0, 13.3, -18.5, 19, 56, -2.6),
    (19, 20, 13.6, -18.7, 19, 41, 13.4, -18.4, 20, 40, -2.5),
    (20, 2, 13.7, -18.6, 20, 24, 13.5, -18.3, 21, 27, -2.4),
    (20, 46, 13.8, -18.5, 21, 10, 13.6, -18.2, 22, 17, -2.3),
    (21, 34, 13.9, -18.4, 21, 59, 13.7, -18.1, 23, 11, -2.2),
    (22, 25, 14.0, -18.3, 22, 52, 13.8, -18.0, 24, 9, -2.1),
    (23, 20, 14.1, -18.2, 23, 49, 13.9, -17.9, 25, 12, -2.0),
    (24, 20, 14.2, -18.1, 24, 51, 14.0, -17.8, 26, 20, -1.9),
    (25, 24, 14.3, -18.0, 25, 58, 14.1, -17.7, 27, 34, -1.8),
    (26, 34, 14.4, -17.9, 27, 11, 14.2, -17.6, 28, 54, -1.7),
    (27, 50, 14.5, -17.8, 28, 31, 14.3, -17.5, 30, 22, -1.6),
    (29, 13, 14.6, -17.7, 29, 58, 14.4, -17.4, 31, 58, -1.5),
    (30, 44, 14.7, -17.6, 31, 33, 14.5, -17.3, 33, 43, -1.4),
    (32, 24, 14.8, -17.5, 33, 18, 14.6, -17.2, 35, 38, -1.3),
    (34, 15, 14.9, -17.4, 35, 15, 14.7, -17.1, 37, 45, -1.2),
    (36, 17, 15.0, -17.3, 37, 24, 14.8, -17.0, 40, 6, -1.1),
    (38, 34, 15.1, -17.2, 39, 48, 14.9, -16.9, 42, 42, -1.0),
    (41, 6, 15.2, -17.1, 42, 28, 15.0, -16.8, 45, 34, -0.9),
    (43, 56, 15.3, -17.0, 45, 29, 15.1, -16.7, 48, 45, -0.8),
    (47, 7, 15.4, -16.9, 48, 52, 15.2, -16.6, 52, 16, -0.7),
    (50, 43, 15.5, -16.8, 52, 41, 15.3, -16.5, 56, 9, -0.6),
    (54, 46, 15.6, -16.7, 56, 59, 15.4, -16.4, 60, 26, -0.5),
    (59, 21, 15.7, -16.6, 61, 50, 15.5, -16.3, 65, 6, -0.4),
    (64, 28, 15.8, -16.5, 67, 15, 15.6, -16.2, 70, 9, -0.3),
    (70, 10, 15.9, -16.4, 73, 14, 15.7, -16.1, 75, 32, -0.2),
    (76, 24, 16.0, -16.3, 79, 42, 15.8, -16.0, 81, 12, -0.1),
    (83, 5, 16.1, -16.2, 86, 31, 15.9, -15.9, 87, 3, 0.0),
)


def corr_tab10(a, summer=0, upper=0):
    assert 0 <= a <= 10, a
    i = 2 + 2 * summer + upper
    t = _alt_corr10
    e0 = t[0]
    a0 = e0[0] + e0[1] / 60
    for k in range(1, len(t)):
        e1 = t[k]
        a1 = e1[0] + e1[1] / 60
        if a0 <= a <= a1:
            c0, c1 = e0[i], e1[i]
            x = (a - a0) / (a1 - a0)
            d = (c1 - c0)
            return c0 + (a - a0) / (a1 - a0) * (c1 - c0)
        e0, a0 = e1, a1


def corr_tab90(a, summer=0, upper=0):
    t = _alt_corr90
    d = 4 * summer
    m = d + 1
    v = d + 2 + upper
    assert t[0][d] + t[0][m] / 60 <= a <= 90, (t[0][d] + t[0][m] / 60, a)
    c = t[0][v]
    for r in t:
        a0 = r[d] + r[m] / 60
        if a < a0: break
        c = r[v]
    return c


def corr_tab(a, summer=0, upper=0):
    try:
        return corr_tab90(a, summer, upper)
    except:
        return corr_tab10(a, summer, upper)


def corr(a, SD=0, tab=1):
    if tab:
        return corr_tab(a, abs(SD) < 16 if SD else 2, SD < 0)
    return SD - 60 * refraction(a, 10, 1010)


def icorr(c, m=0, n=1, *args, **kwargs):
    c = round(c, n)
    for am in arange(0, 90 * 60, 10 ** -m):
        a = round(am, m) / 60
        if round(corr(a, *args, **kwargs), n) >= c:
            return a
    return 90


def ft(m):
    "meters to feet"
    return 3.280839 * m


def dip(h, feet=0):
    "dip in minutes from height in meters"
    return (0.971 if feet else 1.765) * sqrt(h)


def idip(d, n=1, feet=0):
    "min. height in meters for dip in minutes"
    d = round(d, n)
    for h in arange(0, 300, 10 ** -n):
        h = round(h, n)
        if round(dip(h, feet), n) >= d:
            return h


def eqot_img(year, filename=None, size=(16, 9)):
    sns.set_theme(context="notebook", style="whitegrid")
    if filename:
        if isfile(filename): return filename
        use("Agg")
    eqot = []
    xticks = []
    xlabels = []
    for d in Bar(filename or "Equation of Time Graph", max=365, suffix="%(percent)d%% %(eta_td)s").iter(range(365)):
        t = datetime(year, 1, 1) + duration(days=d)
        if t.day in [1, 10, 20]:
            xticks.append(d)
            xlabels.append(f"{t:%b}" if t.day == 1 else t.day)
        eqot.append(equation_of_time(t) * 60)

    figure(figsize=size)
    plot(eqot)
    title(f"Equation of Time {year}")
    xlabel("month of year")
    ylabel("solar time - mean time (UT1) [minutes]")
    xlim(0, 364)
    gca().set_xticks(xticks, xlabels)
    tight_layout()
    if filename:
        savefig(filename, pad_inches=0)
    return filename


def mp_img(year, filename=None, size=(16, 9)):
    sns.set_theme(context="notebook", style="whitegrid")
    if filename:
        if isfile(filename): return filename
        use("Agg")
    dashes = {"Sun": [],
              "Jupiter": [6, 2],
              "Saturn": [1, 2],
              "Venus": [1, 2, 1, 2, 6, 2],
              "Mars": [1, 2, 6, 2, 6, 2],
              "Mercury": [1, 2, 6, 2], }
    mp = {p: [] for p in dashes.keys()}
    xticks = []
    xlabels = []
    for d in Bar(filename or "Meridian Passages Graph", max=365, suffix="%(percent)d%% %(eta_td)s").iter(range(365)):
        t = datetime(year, 1, 1) + duration(days=d)
        if t.day in [1, 10, 20]:
            xticks.append(d)
            xlabels.append(f"{t:%b}" if t.day == 1 else t.day)
        for p in mp.keys():
            mp[p].append(meridian_passage(t, p))

    figure(figsize=size)
    for p, d in mp.items():
        d = array(d)
        d[d > 23.8] = nan
        plot(d, label=p, dashes=dashes[p])
    title(f"Meridian Passages {year}")
    xlabel("month of year")
    ylabel("MP [hours]")
    xlim(0, 364)
    ylim(0, 24)
    legend()
    gca().set_xticks(xticks, xlabels)
    yticks = 0, 3, 6, 9, 12, 15, 18, 21
    gca().set_yticks(yticks, yticks)
    tight_layout()
    if filename:
        savefig(filename, pad_inches=0)
    return filename


def render(template, variables={}, generate=False, progress=None):
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
    env.filters.update({
        "round": round,
        "f": f,
        "dm": dm,
        "hms": hms,
        "hm": hm,
        "ms": ms,
        "rep": replace,
    })
    env.globals.update(variables)
    env.globals.update({
        "today": datetime.utcnow().date(),
        "time": time,
        "datetime": datetime,
        "duration": duration,
        "days": lambda n: duration(days=n),
        "hours": lambda n: duration(hours=n),
        "dut1": delta_ut1,
        "dtt": delta_t,
        "gha_dec": gha_dec,
        "sha_dec": sha_dec,
        "v_value": v_value,
        "d_value": d_value,
        "SD": semi_diameter,
        "hp_moon": hp_moon,
        "mag": magnitude,
        "MP": meridian_passage,
        "eqot": equation_of_time,
        "age": lambda t: moon_age_phase(t)[0],
        "phase": lambda t: moon_age_phase(t)[1],
        "stars": stars,
        "planets": planets,
        "latitudes": latitudes,
        "twilight": twilight,
        "moon_rs": moon_rise_set,
        "eqot_img": lambda *a, **k: eqot_img(*a, **k),
        "mp_img": lambda *a, **k: mp_img(*a, **k),
        "progress": lambda *a, **k: "",
        "decimals": lambda n: str(decimals(n)) * 0,
        "marker": lambda k, m: str(marker(k, m)) * 0,
        "inc_sun": inc_sun,
        "inc_aries": inc_aries,
        "inc_moon": inc_moon,
        "v_corr": v_corr,
        "sqrt": sqrt,
        "dip": dip,
        "idip": idip,
        "corr": corr,
        "icorr": icorr,
        "iers": iers_info,
    })

    if progress:
        progress.start()
        env.globals["progress"] = lambda *a: str(progress.next(*a)) * 0

    template = env.get_template(template)
    return template.generate() if generate else template.render()


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        prog="almanac",
        description="astro navigation tables generator",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("template", help="jinja template to render")
    parser.add_argument("variables", help="set variables name=value", nargs="*")
    parser.add_argument("-o", "--output", metavar="file", help="output file, - for stdout")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite")
    parser.add_argument("-c", "--cache", action="store_true", help="load/save cached values")
    parser.add_argument("-y", "--year", type=int, default=datetime.utcnow().year, help="year to generate data for")
    parser.add_argument("-s", "--start", type=int, default=0, help="offset for start day of year")
    parser.add_argument("-d", "--days", type=int, default=365, help="number of days to generate")
    parser.add_argument("-F", "--no-finals", action="store_true", help="do not use IERS time data (implies -P)")
    parser.add_argument("-P", "--no-polar", action="store_true", help="do not correct for polar motion")
    parser.add_argument("-e", "--ephemeris", metavar="file", default="de440s", help="ephemeris file to use")
    args = parser.parse_args()

    iers_time = not args.no_finals
    polar_motion = iers_time and not args.no_polar

    variables = {
        "year": args.year,
        "odays": args.start,
        "ndays": args.days,
        "iers_time": iers_time,
        "polar_motion": polar_motion,
        "ephemeris": args.ephemeris,
    }

    variables.update({v.split("=", 1)[0]: parse(v.split("=", 1)[1]) for v in args.variables})

    init(iers_time, polar_motion, args.ephemeris, args.cache)

    assert isfile(args.template), args.template + " template not found"

    if args.output and args.output == "-":
        for l in render(args.template, variables=variables, generate=1):
            print(l, end="")
    else:
        out = args.output or args.template.replace(".j2", "")
        assert not isfile(out) or args.force, out + " exists, use -f to overwrite"
        bar = Bar(out, max=args.days, suffix="%(percent)d%% %(eta_td)s")
        with open(out, "w") as f:
            for l in render(args.template, variables=variables, generate=1, progress=bar):
                f.write(l)
        bar.finish()


if 0:
    init()
    for i in range(365):
        t = datetime(2021, 1, 1) + duration(days=i)
        print(f"{i:3}", end=" ")
        for b in "Sun", "Venus", "Mars", "Jupiter", "Saturn":
            print(b[0], mi(v_value(t, b)), mi(d_value(t, b), signed=0), end=" ")
        print()

if 0:
    init()

    import pyinputplus as pyip

    now = datetime0.utcnow()
    d = pyip.inputDate(f"date ({now:%Y-%m-%d}): ",
                       blank=True,
                       formats=("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%y-%m-%d")) or now.date()
    t = pyip.inputTime(f"time ({now:%H:%M:%S}): ", blank=True) or now.time()
    t = datetime0.combine(d, t).replace(tzinfo=utc)
    t = time(t)
    print(dtime(t))
    b = pyip.inputStr("Body (Sun, Moon, Planet, Star): ", blank=True) or "Sun"
    print(b)
    lat = pyip.inputCustom(angle, "Lat: ")
    lon = pyip.inputCustom(angle, "Lon: ")
    print("Lat", dm(lat), "Lon", dm(lon))
    if b in stars:
        sha, dec = sha_dec(t, b)
        print("SHA", dm(sha), "Dec", dm(dec))
        ghaa, _ = gha_dec(t, ARIES)
        print("GHAA", dm(sha))
    gha, dec = gha_dec(t, b)
    print("GHA", dm(gha), "Dec", dm(dec))
    lha, dec = lha_dec(t, b, lon)
    print("LHA", dm(lha), "Dec", dm(dec))
    hc, zn = alt_az(t, b, lat, lon)
    print("Hc ", dm(hc), "Zn ", dm(zn))

    hs = pyip.inputCustom(angle, "Height of Sextant: ")
    print("Hs ", dm(hs))
    he = pyip.inputFloat("Heigt of Eye (m): ", blank=True) or 2
    dipc = -dip(he) / 60
    print("HoE", f"{he:.1f}m", "DIP", mi(dipc))
    ha = hs + dipc
    print("Ha ", dm(ha))
    sd = semi_diameter(t, b) if b == "Sun" else 0
    if sd:
        sd *= -1 if (pyip.inputStr("Limb (L/U): ", blank=True) or "L").upper() == "U" else 1
    acorr = corr(ha, sd * 60) / 60
    print("AC ", mi(acorr), f"SD {mi(sd)}" if sd else "")
    ho = hs + dipc + acorr
    print("Ho ", dm(ho))
    ic = ho - hc
    print("Intercept", dm(ic))

if __name__ == "__main__":
    main()

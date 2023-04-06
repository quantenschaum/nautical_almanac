#!/usr/bin/env python
import atexit
import pickle
from collections import defaultdict
from datetime import date
from math import cos, sqrt, atan, sin, asin, acos, radians, degrees
from multiprocessing import set_start_method, Process, Queue
from os import environ
from queue import Empty
from time import sleep

from cachetools import cached
from skyfield import almanac
from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos, iers
from skyfield.framelib import itrs
from skyfield.magnitudelib import planetary_magnitude
from skyfield.timelib import Time

from corrections import *
from formatting import *
from plots import *

__version__ = "0.2"

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

_cache = defaultdict(dict)


def merge_cache(c):
    for k, v in c.items():
        _cache[k].update(v)


def load_cache(filename="cache.pkl"):
    with open(filename, "rb") as f:
        merge_cache(pickle.load(f))


def dump_cache(filename="cache.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(_cache, f)


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


def init(iers_time=True, polar_motion=True, ephemeris="de440s", cache=None):
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

    if cache and "r" in cache:
        try:
            load_cache()
        except:
            pass

    if cache and "w" in cache:
        atexit.register(dump_cache)


def body(name):
    "return ephemeris for named body"
    return bodies[name]


# @cached({})
def time(t, *a, **k):
    "python datetime (UT1) -> skyfield time"
    if isinstance(t, Time):
        return t
    if isinstance(t, datetime):
        # return ts.from_datetime(t.replace(tzinfo=utc))
        return ts.ut1(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
    return datetime(t, *a, **k)


def dtime(t):
    "skyfield time -> python datetime (UT1, rounded to integer seconds)"
    # return t.utc_datetime().replace(tzinfo=None)
    c = list(t.ut1_calendar())
    c[5] = round(c[5])
    if c[5] >= 60:
        c[5] -= 60
        return datetime(*c) + timedelta(minutes=1)
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


_gha = 0  # 0=radec+gast, 1=latlon(itrs), 2=like 1 with GHAA of 0


# @cached({})
def earth_at(t):
    return earth.at(t)


# @cached({})
def observe(t, b, a=True):
    o = earth_at(t).observe(bodies[b])
    return o.apparent() if a else o


@cached(_cache["sha"])
def sha_dec(t, b):
    "SHA and Dec of body b at time t in degrees"
    if _gha == 0:
        if b == ARIES:
            return 0.0, 0.0
        t = time(t)
        ra, dec, _ = observe(t, b).radec("date")
        return -ra._degrees % 360, dec.degrees
    gha, dec = gha_dec(t, b)
    sha = (gha - gha_dec(t, ARIES)[0]) % 360
    return sha, dec


def gha_dec(t, b, sunvcorr=False):
    gha, dec = _gha_dec(t, b)
    if sunvcorr and b == "Sun":  # add v/2 to SUN's GHA as explained in the NA
        gha = (gha + v_value(t, b) / 120) % 360
    return gha, dec


@cached(_cache["gha"])
def _gha_dec(t, b):
    "GHA and Dec of body b at time t in degrees"
    if _gha == 0 or _gha == 2 and b == ARIES:
        if b == ARIES:
            return 15 * time(t).gast, 0.0  # ICRS right ascension (0) + gast
        sha, dec = sha_dec(t, b)
        gha = (sha + gha_dec(t, ARIES)[0]) % 360
        return gha, dec

    # using frame_latlon instead of radec+gast also aplies time.M and polar_motion_matrix
    # which results in ITRS GHA, Dec but do not match the values from the almanac
    t = time(t)
    lat, lon, _ = observe(t, b).frame_latlon(itrs)
    dec, gha = lat.degrees, -lon.degrees % 360
    return gha, dec


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


@cached(_cache["sd"])
def semi_diameter(t, b):
    "semi diameter of body b at time t in arc minutes"
    t = time(t)
    _, _, dist = observe(t, b).radec(t)
    radius = {"Sun": 695997, "Moon": 1739.9}  # km
    return degrees(atan(radius[b] / dist.km)) * 60


def hp_moon(t):
    "moon's horizontal parallax at time t in arc minutes"
    return semi_diameter(t, "Moon") / 0.272805950305


def d_value(t, b):
    "d value of body b (rate of change of DEC) in arc minutes/hour"
    gha0, dec0 = gha_dec(t, b)
    gha1, dec1 = gha_dec(t + timedelta(hours=1), b)
    dec0, dec1 = dec0 * 60, dec1 * 60
    # dec0, dec1 = round(dec0 * 60, 1) / 60, round(dec1 * 60, 1) / 60
    return abs(dec1) - abs(dec0) if dec0 * dec1 > 0 else (dec1 - dec0)


def v_value(t, b):
    "v values of body b (excess rate of change of GHA) in arc minutes/hour"
    gha0, dec0 = gha_dec(t, b)
    gha1, dec1 = gha_dec(t + timedelta(hours=1), b)
    base = (14 + 19 / 60) if b == "Moon" else 15
    # gha0, gha1 = round(gha0 * 60, 1) / 60, round(gha1 * 60, 1) / 60
    return ((gha1 - gha0) % 360 - base) * 60


@cached(_cache["mag"])
def magnitude(t, b):
    "magnitude of body b at time t"
    t = time(t)
    if b in stars:
        return star_df.loc[stars[b]].magnitude
    else:
        m = planetary_magnitude(observe(t, b))
        return float(m)


@cached(_cache["eqot"])
def equation_of_time(t):
    "equation of time (solar time - UT1) at time t in minutes"
    gha, dec = gha_dec(t, "Sun")
    tsun = (gha / 15 - 12) % 24  # solar time
    tut1 = hours(time(t))
    eqot = tsun - tut1
    if abs(eqot) > 1:
        eqot -= copysign(24, eqot)
    return eqot * 60


@cached(_cache["mp"])
def meridian_passage(t, b, lon=0, upper=True):
    "time of meridian passage of body b at date t in hours"
    f = almanac.meridian_transits(eph, bodies[b], wgs84.latlon(0, lon))
    t0, t1 = time(t), time(t + timedelta(hours=24))
    times, events = almanac.find_discrete(t0, t1, f)
    times = times[events == bool(upper)]
    #     assert len(times) <= 1,(s,t0,t1,times)
    if not times:
        return None
    return hours(times[0])


@cached(_cache["sr"])
def sunrise_sunset(t, lat, lon=0):
    "sunset/sunrise times 1=rise 0=set"
    t0, t1 = time(t), time(t + timedelta(hours=24))
    f = almanac.sunrise_sunset(eph, wgs84.latlon(lat, lon))
    times, events = almanac.find_discrete(t0, t1, f)
    f0 = int(f(t0))
    data = {"t0": f0, "min": f0, "max": f0}
    for t, e in zip(times, events):
        data[e] = hours(t)
        data["min"] = min(data["min"], e)
        data["max"] = max(data["max"], e)
    return data


@cached(_cache["tw"])
def twilight(t, lat, lon=0):
    """twilights and sunset/sunrise
    0 = night, dark
    1 = astronomical twilight
    2 = nautical twilight
    3 = civil twilight
    4 = day, sun is up
    negative sign, if sun is going down
    """
    t0, t1 = time(t), time(t + timedelta(hours=24))
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


@cached(_cache["mr"])
def moon_rise_set(t, lat, lon=0):
    "time of moon rise and set"
    s = "Moon"
    sd = semi_diameter(t, s) / 60
    f = almanac.risings_and_settings(eph, bodies[s], wgs84.latlon(lat, lon), radius_degrees=sd)
    f.step_days = 0.01
    t0, t1 = time(t), time(t + timedelta(hours=24))
    times, events = almanac.find_discrete(t0, t1, f)
    f0 = int(f(t0))
    data = {"t0": f0, "min": f0, "max": f0}
    for t, e in zip(times, events):
        data[e if e not in data else e + 10] = hours(t)
        data["min"] = min(data["min"], e)
        data["max"] = max(data["max"], e)
    return data


@cached(_cache["ma"])
def moon_age_phase(t):
    "moon age (time since new moon) in days and phase (fraction illuminated)"
    t0, t1 = time(t - timedelta(days=30)), time(t + timedelta(hours=24))
    p = almanac.moon_phase(eph, time(t)).radians
    illum = (1 - cos(p)) / 2
    times, events = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
    new_moon = times[events == 0][0]
    age = t - dtime(new_moon)
    age = age.total_seconds() / 3600 / 24
    return age, illum


def render(template, variables={}, progress=None):
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
    env.filters.update({
        "round": round,
        "f": f,
        "dm": dm,
        "hms": hms,
        "hm": hm,
        "ms": hm,
        "rep": replace,
    })
    env.globals.update(variables)
    env.globals.update({
        "now": datetime.utcnow(),
        "today": datetime.utcnow().date(),
        "time": time,
        "duration": timedelta,
        "days": lambda n: timedelta(days=n),
        "hours": lambda n: timedelta(hours=n),
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
        "print": lambda *a, **k: str(print(*a, **k)) * 0,
    })

    if progress:
        env.globals["progress"] = lambda *a: str(progress(a[0] if a else 1)) * 0
    else:
        env.globals["progress"] = lambda *a: ""

    template = env.get_template(template)
    return template.generate()


def calculate():
    import pyinputplus as pyip

    now = datetime.utcnow()
    d = pyip.inputDate(f"date ({now:%Y-%m-%d}): ",
                       blank=True,
                       formats=("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%y-%m-%d")) or now.date()
    t = pyip.inputTime(f"time ({now:%H:%M:%S}): ", blank=True) or now.time()
    t = datetime.combine(d, t)
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
    dipc = -dip(he)
    print("HoE", f"{he:.1f}m", "DIP", f(dipc))
    ha = hs + dipc / 60
    print("Ha ", dm(ha))
    sd = semi_diameter(t, b) if b == "Sun" else 0
    if sd:
        sd *= -1 if (pyip.inputStr("Limb (L/U): ", blank=True) or "L").upper() == "U" else 1
    acorr = corr(ha, sd)
    print("AC ", f(acorr), f"SD {f(sd)}" if sd else "")
    ho = ha + acorr / 60
    print("Ho ", dm(ho))
    ic = (ho - hc)
    print("Intercept", dm(ic))


DEVNULL = "/dev/null"


def process(template, out, variables, progress=None):
    # for i in range(variables["ndays"]):
    #    sleep(1)
    #    progress(1)
    # return
    init(variables["iers_time"], variables["polar_motion"], variables["ephemeris"], variables["cache"])
    if out == "-":
        for l in render(template, variables, progress):
            print(l, end="")
    elif out == DEVNULL:
        for l in render(template, variables, progress):
            pass
    else:
        with open(out, "w") as f:
            for l in render(template, variables, progress):
                f.write(l)

    if variables.get("push_cache"):
        progress(_cache)


def parallel(args, variables):
    set_start_method('spawn')
    n = args.parallel
    w = args.days // n
    m = args.multiple
    w += m - w % m if w % m else 0  # make multiple of 3 because pages contain 3 days
    assert not w % m, w
    l = max(0, args.days - (n - 1) * w)  # last segment
    processes = []
    k = Queue()
    variables["push_cache"] = True
    variables["cache"] = "r" if args.cache else None
    bar = Bar(f"computing with {n} processes", max=args.days,
              suffix="%(percent)d%% %(eta_td)s %(index)s %(elapsed_td)s")
    bar.start()
    for v in "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS":
        if v not in environ:
            environ[v] = "1"
    for i in range(n):
        variables["odays"] = args.start + i * w
        variables["ndays"] = w if i < n - 1 else l
        p = Process(target=process, args=(args.template, DEVNULL, variables, k.put_nowait))
        processes.append(p)
        p.start()
    while any(map(Process.is_alive, processes)):
        try:
            n = k.get_nowait()
            if isinstance(n, dict):
                merge_cache(n)
            else:
                bar.next(n)
        except Empty:
            pass
        sleep(0.1)
    bar.finish()
    variables["push_cache"] = False
    variables["cache"] = "w" if args.cache else None
    variables["odays"] = args.start
    variables["ndays"] = args.days


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        prog="almanac",
        description="astro navigation tables generator " + __version__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("template", help="jinja template to render", nargs="?")
    parser.add_argument("-o", "--output", metavar="file", help="output file, - for stdout")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite")
    parser.add_argument("-c", "--cache", action='store_const', const="rw", help="load/save cached values")
    parser.add_argument("-y", "--year", type=int, default=datetime.utcnow().year, help="year to generate data for")
    parser.add_argument("-s", "--start", type=int, default=0, help="offset for start day of year")
    parser.add_argument("-d", "--days", type=int, default=365, help="number of days to generate")
    parser.add_argument("-S", "--set", help="set variables with name=value", action="append")
    parser.add_argument("-F", "--no-finals", action="store_true", help="do not use IERS time data (implies -P)")
    parser.add_argument("-P", "--no-polar", action="store_true", help="do not correct for polar motion")
    parser.add_argument("-e", "--ephemeris", metavar="file", default="de440s", help="ephemeris file to use")
    parser.add_argument("-C", "--calculate", action="store_true", help="interactive sight reduction calculation")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="number of parallel processes to use")
    parser.add_argument("-m", "--multiple", type=int, default=3, help="number of parallel processes to use")
    parser.add_argument("-V", "--version", action="version", version=__version__)
    args = parser.parse_args()

    iers_time = not args.no_finals
    polar_motion = iers_time and not args.no_polar

    if args.calculate:
        init(iers_time, polar_motion, args.ephemeris, args.cache)
        calculate()
        return

    assert args.template, "no template"
    assert isfile(args.template), args.template + " template not found"

    variables = {
        "year": args.year,
        "odays": args.start,
        "ndays": args.days,
        "iers_time": iers_time,
        "polar_motion": polar_motion,
        "ephemeris": args.ephemeris,
        "cache": args.cache,
    }

    if args.set:
        variables.update({v.split("=", 1)[0]: parse(v.split("=", 1)[1]) for v in args.set})

    out = args.output or args.template.replace(".j2", "")
    assert not isfile(out) or args.force, f"{out} exists, use -f to overwrite"

    if args.parallel > 1:
        parallel(args, variables)

    bar = Bar(out, max=args.days, suffix="%(percent)d%% %(eta_td)s %(index)s %(elapsed_td)s")
    bar.start()
    process(args.template, out, variables, bar.next)
    bar.finish()


if __name__ == "__main__":
    main()

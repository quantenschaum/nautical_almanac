import re
from datetime import datetime, timedelta
from os import listdir

from almanac import parse, gha_dec, init, sha_dec, dm, magnitude, v_value, d_value, hp_moon, equation_of_time, \
    semi_diameter, meridian_passage, moon_age_phase, hm


def count_hours(filename):
    hours = 0
    with open(filename) as f:
        for line in f:
            if "|" in line:
                cols = [c.strip() for c in line.strip().split("|")]
                cols = list(filter(len, cols))
                if isinstance(parse(cols[0]), int):
                    hours += 1
    return hours


def read_page(filename):
    hours = count_hours(filename)
    data = []
    with open(filename) as f:
        for line in f:
            if "dUT1" in line:
                mode = None
                date = line.split("dUT1")[0].strip()
                t0 = datetime.strptime(date, "%Y %B %d (%a)")
                t1 = t0 + timedelta(days=int(hours / 24 / 2))
                t2 = t0 + timedelta(hours=hours / 2)
            elif "|" in line:
                cols = [c.strip() for c in line.strip().split("|")]
                cols = list(filter(len, cols))
                # print(cols)
                if cols[0] == "UT":
                    mode = "GHA"
                    hour = 0
                    names = []
                    for c in cols:
                        name_mag = c.split()
                        name = name_mag[0]
                        names.append(name)
                        if len(name_mag) == 2:
                            data.append([t2, name, "mag", parse(name_mag[1])])
                elif cols[0] == "Star":
                    mode = "SHA"
                elif mode == "GHA" and isinstance(parse(cols[0]), int):
                    for i, c in enumerate(cols):
                        name = names[i]
                        # print(name, c)
                        if name == "UT":
                            th = t0 + timedelta(hours=hour)
                            assert th.hour == parse(c) % 24, (th, c)
                            hour += 1
                        elif name == "Aries":
                            data.append([th, name, "GHA", parse(c)])
                        elif name == "Moon":
                            c = c.split()
                            if len(c) == 7:
                                data.append([th, name, "GHA", parse(f"{c[0]} {c[1]}")])
                                data.append([th, name, "v", parse(c[2])])
                                data.append([th, name, "Dec", parse(f"{c[3]} {c[4]}")])
                                data.append([th, name, "d", parse(c[5])])
                                data.append([th, name, "HP", parse(c[6])])
                            elif len(c) == 4:
                                data.append([th, name, "GHA", parse(f"{c[0]} {c[1]}")])
                                data.append([th, name, "Dec", parse(f"{c[2]} {c[3]}")])
                            else:
                                assert 0, c
                        else:
                            c = c.split()
                            assert len(c) == 4, c
                            data.append([th, name, "GHA", parse(f"{c[0]} {c[1]}")])
                            data.append([th, name, "Dec", parse(f"{c[2]} {c[3]}")])
                elif mode == "SHA" and len(cols) == 2:
                    name = cols[0]
                    c = cols[1].split()
                    data.append([t2, name, "SHA", parse(f"{c[0]} {c[1]}")])
                    data.append([t2, name, "Dec", parse(f"{c[2]} {c[3]}")])
                elif mode == "GHA":
                    for i, c in enumerate(cols):
                        if c.startswith("SD") or c.startswith("v"):
                            v = c.split()
                            for k in range(0, len(v), 2):
                                data.append([t2, names[i + 1 + (1 if i > 1 else 0)], v[k], float(v[k + 1])])
                        elif c.startswith("EoT"):
                            v = c.split()
                            data.append([t1, names[i + 1], v[0], parse(v[1]) / 60])
                            data.append([t1 + timedelta(hours=12), names[i + 1], v[0], parse(v[2]) / 60])
                        elif c.startswith("Age"):
                            v = c.split()
                            data.append([t2, names[i + 1], v[0], int(v[1][:-1])])
                            data.append([t2, names[i + 1], v[2], int(v[3][:-1])])
                        elif c.startswith("Upper"):
                            v = c.split()
                            data.append([t1, names[i + 1], v[0], parse(v[1])])
                            data.append([t1, names[i + 1], v[2], parse(v[3])])
                        elif c.startswith("MP"):
                            v = c.split()
                            data.append([t1, names[i + 1], v[0], parse(v[1])])
                        elif ":" in c:
                            data.append([t1, names[i + 1], "MP", parse(c)])
                        elif i > 2:
                            v = parse(c)
                            if isinstance(v, float):
                                data.append([t1, names[i + 1], "SHA", v])

    return data


def compare(filename):
    data = read_page(filename)
    init()
    diff = {}
    for r in data:
        t, b, n, v = r
        if n == "GHA":
            w = gha_dec(t, b)[0]
        elif n == "Dec":
            w = gha_dec(t, b)[1]
        elif n == "SHA":
            w = sha_dec(t, b)[0]
        elif n == "mag":
            w = magnitude(t, b)
        elif n == "v":
            w = v_value(t, b)
        elif n == "d":
            w = d_value(t, b)
        elif n == "HP":
            assert b == "Moon", b
            w = hp_moon(t)
        elif n == "EoT":
            assert b == "Sun", b
            w = equation_of_time(t)
        elif n == "SD":
            w = semi_diameter(t, b)
        elif n in ["MP", "Upper", "Lower"]:
            w = meridian_passage(t, b, upper=n != "Lower")
        elif n == "Age":
            assert b == "Moon", b
            w = moon_age_phase(t)[0]
        elif n == "Phase":
            assert b == "Moon", b
            w = moon_age_phase(t)[1] * 100
        else:
            print(r)
            continue
        if n in ["GHA", "SHA", "Dec"]:
            w = parse(dm(w))
        elif n in ["EoT"]:
            w = parse(hm(w)) / 60
        elif n in ["MP", "Upper", "Lower"]:
            w = parse(hm(w))
        elif n in ["Age", "Phase"]:
            w = round(w)
        else:
            w = round(w, 1)
        d = abs(w - v)
        if n in ["GHA", "SHA", "Dec", "MP", "Upper", "Lower"]:
            d *= 60
        mm = diff.setdefault(b, {}).setdefault(n, [0, 0, 0])
        aa = "AA" in filename  # AirAlmanac
        if not (b == "Moon" and aa):
            assert d <= (0.25 if n in ["GHA", "Dec"] else 1), (filename, r, w, d)
        mm[0] = max(mm[0], d)
        mm[1] += 1 if d > 0 else 0
        mm[2] += 1

    diff2 = []
    for b, v in diff.items():
        for k, v in v.items():
            diff2.append([b, k] + v)

    print(filename)
    r = "body", "value", "maxdev", "dev", "tot"
    # print(f"{r[0]:15} {r[1]:6} {r[2]:6} {r[3]:>6}/{r[4]}")
    for r in diff2:
        if r[2] and r[0] == "Sun":
            print(f"{r[0]:15} {r[1]:6} {r[2]:6.4f} {r[3]:6}/{r[4]}")


def main():
    for d in listdir():
        if re.match("daily-pages-\\d{4}-.*.txt", d):
            compare(d)


if __name__ == "__main__":
    main()

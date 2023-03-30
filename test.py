from datetime import datetime, timedelta

from almanac import parse, gha_dec, init, sha_dec, marker, dm, magnitude, v_value, d_value, hp_moon


def read_page(filename):
    data = []
    with open(filename) as f:
        for line in f:
            if "dUT1" in line:
                date = line.split("dUT1")[0].strip()
                t0 = datetime.strptime(date, "%Y %B %d (%a)")
                t1 = t0 + timedelta(hours=24)
                t2 = t0 + timedelta(hours=36)
            if "|" in line:
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
                            assert len(c) == 7, c
                            data.append([th, name, "GHA", parse(f"{c[0]} {c[1]}")])
                            data.append([th, name, "v", parse(c[2])])
                            data.append([th, name, "Dec", parse(f"{c[3]} {c[4]}")])
                            data.append([th, name, "d", parse(c[5])])
                            data.append([th, name, "HP", parse(c[6])])
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

    return data


def compare(filename):
    data = read_page(filename)
    init()
    diff = {}
    marker("°", " ")
    marker("'", "")
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
            w = v_value(t, b) * 60
        elif n == "d":
            w = d_value(t, b) * 60
        elif n == "HP":
            assert b == "Moon", b
            w = hp_moon(t) * 60
        else:
            continue
        f = 60 if n in ["GHA", "SHA", "Dec"] else 1
        w = parse(dm(w)) if n in ["GHA", "SHA", "Dec"] else round(w, 1)
        d = (w - v) * f
        mm = diff.setdefault(b, {}).setdefault(n, [0, 0, 0])
        assert abs(d) < 0.21, (r, w, d)
        if d != 0:
            mm[2] += 1
        for i, g in enumerate((min, max)):
            mm[i] = g(mm[i], d)

    diff2 = []
    for b, v in diff.items():
        for k, v in v.items():
            diff2.append([b, k, "min", v[0]])
            diff2.append([b, k, "max", v[1]])
            diff2.append([b, k, "abs", max(-v[0], v[1])])
            diff2.append([b, k, "n", v[2]])

    print(data[0][0])
    total = 0
    for r in diff2:
        if r[3] and r[2] in ["abs", "n"]:
            if r[2] == "n":
                total += r[3]
            print(f"{r[0]:10} {r[1]:3} {r[2]:3} {r[3]}")
    print(f"           total   {total}\n")


compare("daily-pages-2021-01-01.txt")
compare("daily-pages-2021-04-01.txt")

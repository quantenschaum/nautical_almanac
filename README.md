# Nautical Almanac

This project was inspired by

- https://thenauticalalmanac.com/
- https://github.com/rodegerdts/Pyalmanac
- https://github.com/aendie/SkyAlmanac-Py3

Thank you for your work!

I started this project because I

- was looking into celestial navigation and searching for a free almanac
- wanted to write a program to create an almanac myself, just for fun
- try out and learn Python and related libs

## Idea

The idea was to create routines that perform

- the actual calculations,
- the formatting of the numbers and
- their compilation into some form of tables.

All three steps should be separated from each other.

And finally the computed data should be validated against some reliable reference like a commercially available paper Almanac.

## Design and Usage

This software currently is a single Python script `almanac.py`. It contains routines for computing GHA, Dec, etc. as they are tabulated in the almanac. The actual celestial equations are handled by the fantastic [Skyfield](https://rhodesmill.org/skyfield/) (Many thanks this one, too). Additionally, there are some helper functions for formatting numbers and such.

The compilation of data tables containing the values is done using the [Jinja](https://jinja.palletsprojects.com/) template engine. This allows the data to be formatted in any format you like. By using templates computation and presentation of the data get completely separated, and it is much easier to maintain and change the output format.

The output values of the SkyField based routines, which take quite some time to evaluate, are cached. The cache may be stored to disk and can be used in future runs to speed up the template rendering.

The script also exposes a command line interface using `argparse` which allows to render a Jinja template and set various parameters. Just run `./almanac.py -h` to see what it can do.

## Templates

Currently, there are 2 templates in the package.

`nautical-almanac.tex.j2` produces LaTeX source code that can further be compiled into a PDF. This is essentially the same output as [SkyAlmanac](https://github.com/aendie/SkyAlmanac-Py3) produces and should contain all necessary information as in the official almanacs. It contains

- Chart of the sky (taken from [SkyAlmanac](https://github.com/aendie/SkyAlmanac-Py3))
- Plots of equation of time and meridian passages of the Sun and the planets
- Arc to Time conversion table
- Altitude correction tables (values from commercial almanac or by [formula](https://en.wikipedia.org/wiki/Atmospheric_refraction#Calculating_refraction))
- Index to selected stars
- Daily pages with GHA, Dec, ...
- Increments and Corrections

There is a `makefile` for this job.

`daily-pages.txt.j2` produces plain text tables with GHA, Dec, ... as on the daily pages. These tables can be used to compare (`diff`) the computed values to reference data or data computed with changed settings or a different implementation. They can be used for diagnostic purposes.

## Validation

I have a copy of the Paracay Nautical Almanac 2021 and some pages of other years are available online

- [Wikipedia](https://en.wikipedia.org/wiki/Nautical_almanac)
- [Training NA 1981 A](https://maritimesafetyinnovationlab.org/wp-content/uploads/2015/01/nautical-almanac-1981.pdf), [Training NA 1981 B](https://www.dco.uscg.mil/Portals/9/NMC/pdfs/examinations/10_1981_nautical_almanac.pdf), [Training NA 1981 C](http://fer3.com/arc/imgx/Nautical-Almanac-1981-(ReedNavigation.com-edit).pdf)
- [AirAlmanac](https://aa.usno.navy.mil/downloads/publications/aira23_all.pdf)

The `daily-pages-yyyy-....txt` files contain data from these sources for reference and can automatically be compared to calculated values using `test.py` (WIP).

Currently, the computed values of GHA/SHA and Dec agree with those published in the commercial almanac within 0.1'. GHA of Aries matches exactly, stars, Moon and planets match exactly for the majority of the values.

The values also agree with those computed by SkyAlmanac (the routines are effectively the same).

### GHA of Sun

The GHA of the Sun is treated differently in the Nautical Alamanac.

The explanations in the NA, in section "[Main Data](exp1.png)" and "[Accuracy](exp2.png)" say

> v for the Sun is negligible and is omitted

> The quantities tabulated in this Almanac are generally correct to the nearest 0.1'; the exception is the Sun's GHA which is deliberately adjusted by up to 0.15' to reduce the error due to ignoring the v-correction.

This explains the systematic difference of the GHA of the Sun. Adding v/2 to the GHA makes this offset disappear. In `test.py` this correction is applied when comparing with data from the NA. This correction can be enabled by setting `sunvcorr=1` when generating tables.

In the AirAlamanac this correction is not applied.

IMHO this does not make much sense. They say v is "negligible" and omit it in the NA, but then they "deliberately adjust" the GHA to "reduce the error due to ignoring v". This is a contradiction. On the other hand, adding v/2 produces correct GHAs for times close to hh:30 but it's off at hh:00, without this correction GHA would be correct at hh:00, but off at hh:30. So, it just shifts the error but does not remove it and the tabulated values for integer hours are off. It would have been easier to just print this single v value as it is done for the planets and list unchanged GHAs. Even without v printed one could interpolate between two GHAs directly (calculating v from the tabulated GHAs).

## GHA and Dec

There are several ways to calculate GHA, SHA and Dec, because there are different coordinate systems.

0. GHA = SHA + GHAA with GHAA = GAST*15 (Greenwich Apparent Siderial Time, GHA of Aries) and SHA = -RA (ICRS right ascension) `radec('date')`
1. GHA = ITRS Longitude and SHA = GHA-GHAA with GHAA = ITRS Longitude of Aries `frame_latlon(itrs)` (includes polar motion)
2. Same as 1, but with GHAA as in 0

GHA and Dec are almost identical for all 3 methods, but GHAA and thus SHA are different for 1. The differences between 0 and 2 are due to polar motion (if enabled with `iers.install_polar_motion_table()`), which currently does not affect the values in the output tables, because the differences are <0.1'. The Nautical Almanac uses a GHAA as in 0.

```
2023-01-01 13:00:00 UT1
Object      M   GHA          Dec          SHA          Dec
Aries       0 295.92252658   0.00000000   0.00000000   0.00000000 
Aries       1 295.63149837   0.12650040   0.00000000   0.12650040 
Aries       2 295.92252658   0.00000000   0.00000000   0.00000000 
Sun         0  14.13837499 -22.99598847  78.21584841 -22.99598847 
Sun         1  14.13835372 -22.99601850  78.50685535 -22.99601850 
Sun         2  14.13835372 -22.99601850  78.21582713 -22.99601850 
Vega        0  16.50241883  38.80390619  80.57989225  38.80390619 
Vega        1  16.50245810  38.80387411  80.87095973  38.80387411 
Vega        2  16.50245810  38.80387411  80.57993151  38.80387411 
```

## Multiprocessing

Multiprocessing is implemented as follows; The range of days is split into n chunks, then the template is rendered for each of the chunks in its own process. The output of these rendering passes is discarded, they are just used to populate the cache. Then the template is rendered again in a final pass over the full range of days. This pass is very fast, since the values get just pulled from the cache.

Unfortunately multiprocessing has not the desired effect of reducing the compute time, because SkyField already uses all cores internally.
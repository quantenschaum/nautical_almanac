# Nautical Almanac

This project was inspired by

- https://thenauticalalmanac.com/
- https://github.com/rodegerdts/Pyalmanac
- https://github.com/aendie/SkyAlmanac-Py3 (I stole some stuff from there)

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

The output values of the SkyField based routines, which take quite some time to evaluate, are cached. The cache may be store to disk and can be used in future runs to speed up the template rendering.

The script also exposes a command line interface using `argparse` which allows to render a Jinja template and set various parameters. Just run `./almanac.py -h` to what it can do.

## Templates

Currently, there are 2 templates in the package.

`nautical-almanac.tex.j2` produces LaTeX source code that can further be compiled into a PDF. This is essentially the same output as [SkyAlmanac](https://github.com/aendie/SkyAlmanac-Py3) produces and should contain all necessary information as in the official almanacs. It contains

- Chart of the sky (taken from [SkyAlmanac](https://github.com/aendie/SkyAlmanac-Py3))
- Plots of equation of time and meridian passages of the sun and the planets
- Arc to Time conversion table
- Altitude correction tables (values from commercial almanac or by [formula](https://en.wikipedia.org/wiki/Atmospheric_refraction#Calculating_refraction))
- Index to selected stars
- Daily paged with GHA, Dec, ...
- Increments and Corrections

There is a `makefile` for this job.

`daily-pages.txt.j2` produces plain text tables with GHA, Dec, ... as on the daily pages. These tables can be used to compare (`diff`) the computed values to reference data or data computed with changed settings or a different implementation. They can be used for diagnostic purposes.

## Validation

I own a copy of the Paracay Nautical Almanac 2021 and some preview pages other year are available online. The `daily-pages-yyyy-mm-dd.txt` files contain data from these sources for reference and can automatically be compared to calculated values using `test.py` (WIP).

Currently, the computed values of GHA/SHA and Dec agree with those published in the commercial almanac within 0.1' except the GHA of the sun which is up to 0.2' off (unclear why). GHA of Aries and SHA And Dec of the Stars match exactly.
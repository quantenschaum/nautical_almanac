Y=$(shell date +%Y)

pdf: Nautical-Almanac-$(Y).pdf
txt: daily-pages-$(Y).txt

Nautical-Almanac-$(Y).pdf: nautical-almanac.tex.j2 almanac.py
	./almanac.py $< -o $(@:.pdf=.tex) -f -c -y$(Y)
	latexmk -pdf $(@:.pdf=.tex) -interaction=batchmode

daily-pages-$(Y).txt: daily-pages.txt.j2 almanac.py
	./almanac.py $< -o $@ -f -y$(Y)

loop:
	while true; do inotifywait -q -e close_write *.j2 *.py makefile; make; done

clean:
	latexmk -C
	rm -f *.tex

cleaner:
	rm -f *.bsp *.all *.dat
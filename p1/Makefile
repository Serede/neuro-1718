.PHONY: all mp_help mp_exec per_help per_exec ada_help ada_exec

all: mp_help per_help ada_help

mp_help:
	python mp.py -h

mp_exec:
	python mp.py -i data/McCulloch_Pitts.txt -o output.txt

per_help:
	python per.py -h

per_exec:
	python per.py -t 10 -l 0.5 -e 100 mode2 -d data/problema_real1.txt

ada_help:
	python per.py -h

ada_exec:
	python ada.py -t 10 -l 0.25 -e 1000 mode2 -d data/problema_real1.txt
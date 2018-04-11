MAIN = p2.py

.PHONY: all retro_help retro_exec retro_exec_6

all: retro_help

retro_help:
	python $(MAIN) -h

retro_exec:
	python $(MAIN) -s 2 2 -i 1 -l 0.25 -e 100 mode1 -d .\data\problema_real2.txt -r 0.7

retro_exec_6:
	python $(MAIN) -s 2 -i 1 -l 0.25 -e 500 mode1 -d .\data\problema_real6.txt -r 0.7
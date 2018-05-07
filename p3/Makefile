PYTHON = python3
AUTOENCODER = p3_autoencoder.py
SERIES = p3_series.py

.PHONY: all autoencoder_help autoencoder_exec series_help series_exec

all: autoencoder_help series_help

autoencoder_help:
	$(PYTHON) $(AUTOENCODER) -h

autoencoder_exec:
	$(PYTHON) $(AUTOENCODER) -s 10 -i 1 -l 0.25 -e 100 mode2 -d data/alfabeto_0e_1c.txt

serie_help:
	$(PYTHON) $(SERIES) -h

serie_exec:
	$(PYTHON) $(SERIES) -s 2 -i 0.5 -l 0.1 -e 2000 mode1 -d data/serie1_2_2.txt -r 0.5
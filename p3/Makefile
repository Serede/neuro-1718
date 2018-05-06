PYTHON = python3
AUTOENCODER = p3_autoencoder.py
SERIES = p3_series.py

.PHONY: all autoencoder_help autoencoder_exec series_help series_exec

all: autoencoder_help series_help

autoencoder_help:
	$(PYTHON) $(AUTOENCODER) -h

autoencoder_exec:
	$(PYTHON) $(AUTOENCODER) -s 10 -i 1 -l 0.25 -e 100 mode2 -d data/alfabeto_2e_7c.txt

series_help:
	$(PYTHON) $(SERIES) -h

series_exec:
	$(PYTHON) $(SERIES) -s 2 -i 1 -l 0.25 -e 100 mode1 -d data/serie1_3_1.txt -r 0.7
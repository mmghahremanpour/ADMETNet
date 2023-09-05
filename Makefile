#Makefile for running ADMETNet program

install:
	python setup.py install

clean:
	rm -rf build ADMETNet.egg-info dist

test:
	solnet -if example/test.csv -of example/test_predictions.csv -nt GCN -dbp pretrained/logS/GCN/HyperParameters-Database.json



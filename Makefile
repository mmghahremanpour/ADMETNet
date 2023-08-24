#Makefile for running ADMETNet program

install:
	python setup.py install

clean:
	rm -rf build ADMETNet.egg-info dist


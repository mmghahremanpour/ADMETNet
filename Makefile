#Makefile for running ADMETNet program

install:
	python setup.py install

clean:
	/usr/bin/rm -rf build ADMETNet.egg-info dist


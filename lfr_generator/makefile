CC=g++
LOP=-o
LOPT=-O3 -funroll-loops

MAIN=./Sources/benchm
TAG=benchmark


$(MAIN).o :
	$(CC) $(LOPT) $(LOP) $(TAG) $(MAIN).cpp

clean :
	rm -f benchmark *.dat

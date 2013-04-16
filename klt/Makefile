all: klt.out

.PHONY: testlu
testlu: testlu.out

CC = g++
CFLAGS = -c -Wall
EXTLIBS = -lopencv_core -lopencv_highgui

klt.out: lu.o proj_align.o main.o
	$(CC) -o klt.out lu.o proj_align.o main.o $(EXTLIBS)

main.o: main.cpp proj_align.h utils.h
	$(CC) $(CFLAGS) main.cpp

proj_align.o: proj_align.h proj_align.cpp utils.h
	$(CC) $(CFLAGS) proj_align.cpp

lu.o: lu.h lu.cpp
	$(CC) $(CFLAGS) lu.cpp

testlu.o: testlu.cpp
	$(CC) $(CFLAGS) testlu.cpp

testlu.out: lu.o testlu.o
	$(CC) -o testlu.out testlu.o lu.o

.PHONY: clean
clean:
	rm *.o klt.out

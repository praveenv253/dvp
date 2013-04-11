all: klt

CC = g++
CFLAGS = -c -Wall
EXTLIBS = -lopencv_core -lopencv_highgui

klt: lu.o proj_align.o main.o
	$(CC) -o klt lu.o proj_align.o main.o $(EXTLIBS)

main.o: main.cpp proj_align.h utils.h
	$(CC) $(CFLAGS) main.cpp

proj_align.o: proj_align.h proj_align.cpp utils.h
	$(CC) $(CFLAGS) proj_align.cpp

lu.o: lu.h lu.cpp
	$(CC) $(CFLAGS) lu.cpp

.PHONY: clean
clean:
	rm *.o klt

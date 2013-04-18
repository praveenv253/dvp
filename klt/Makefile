all: klt.out

.PHONY: testlu
testlu: testlu.out

CC = g++
CFLAGS = -c -Wall
EXTLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc

klt.out: lu.o proj_align.o main.o stitch.o
	$(CC) -o klt.out lu.o proj_align.o stitch.o main.o $(EXTLIBS)

main.o: main.cpp proj_align.h utils.h stitch.h
	$(CC) $(CFLAGS) main.cpp

proj_align.o: proj_align.h proj_align.cpp utils.h
	$(CC) $(CFLAGS) proj_align.cpp

stitch.o: stitch.h stitch.cpp utils.h
	$(CC) $(CFLAGS) stitch.cpp

lu.o: lu.h lu.cpp
	$(CC) $(CFLAGS) lu.cpp

testlu.o: testlu.cpp
	$(CC) $(CFLAGS) testlu.cpp

testlu.out: lu.o testlu.o
	$(CC) -o testlu.out testlu.o lu.o

.PHONY: clean
clean:
	rm *.o klt.out

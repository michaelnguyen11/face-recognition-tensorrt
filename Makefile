all: build

clean:
	rm -rf build

build: clean
	mkdir build
	cd build/ && cmake .. && make -j4

debug: clean
	mkdir build
	cd build/ && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j4

release: clean
	mkdir build
	build/ && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4

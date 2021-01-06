all: fast_build

clean:
	rm -rf build

fast_build: clean
	mkdir build && cd build && cmake .. && make -j4

gpu_degalate: clean
	mkdir build && cd build && cmake -DGPU_DELEGATE=ON .. && make -j4

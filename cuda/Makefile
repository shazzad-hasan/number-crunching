CXX = g++
CXXFLAGS = -O3 -march=native
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60

number_crunching: number_crunching.cpp
	$(CXX) $(CXXFLAGS) -o number_crunching number_crunching.cpp

loopgpu: number_crunching_loop.cu
	$(NVCC) $(NVCCFLAGS) -o number_crunching_loop number_crunching_loop.cu

taskgpu:
	$(NVCC) $(NVCCFLAGS) -o number_crunching_task number_crunching_task.cu

clean:
	rm -f number_crunching number_crunching_loop number_crunching_task

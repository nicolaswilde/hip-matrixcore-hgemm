default: compile run

compile:
	/opt/rocm-6.3.1/bin/hipcc --offload-arch=gfx942 -std=c++20 -O3 -L/opt/rocm-6.3.1/lib -lhipblas  test.hip -o test

run:
	./test
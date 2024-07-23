PY_INC=$(shell python3 -c "import sysconfig;print(sysconfig.get_paths()['platinclude'])")
NP_INC=$(shell python3 -c "import numpy;print(numpy.get_include())")

all: test_driver

test_driver: test_driver.c
	gcc -o test_driver test_driver.c -I${PY_INC} -I${NP_INC} -lpython3.10

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL dsm
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include<stdio.h>
#include<stdlib.h>

static void *bootstrap_python()
{
    char *pypath = getenv("PYTHONPATH");
    char *new_pypath;
    int pypath_len;

    pypath_len = 2;
    if(pypath) {
        pypath_len += strlen(pypath) + 1;
    }
    new_pypath = malloc(pypath_len);
    if(pypath) {
        sprintf(new_pypath, ".:%s", pypath);
    } else {
        strcpy(new_pypath, ".");
    }
    setenv("PYTHONPATH", new_pypath, 1);

    Py_Initialize();
    import_array();
}

int main(int argc, char *argv[])
{
    PyObject *pName, *pMod, *pFunc, *pResult;

    if(argc != 2) {
        fprintf(stderr, "Usage: %s <module>\n", argv[0]);
        goto err;
    }

    bootstrap_python();

    pName = PyUnicode_DecodeFSDefault(argv[1]);
    pMod = PyImport_Import(pName);
    if(!pMod) {
        fprintf(stderr, "ERROR: could not load module '%s'\n", argv[1]);
        goto err_pyprint;
    }
    Py_DECREF(pName);

    pFunc = PyObject_GetAttrString(pMod, "test_driver");
    if(!pFunc || !PyCallable_Check(pFunc)) {
        fprintf(stderr, "ERROR: test_driver() either does not exist or is not a function in %s.\n", argv[1]);
        goto err;
    }

    pResult = PyObject_CallNoArgs(pFunc);
    if(!pResult) {
        fprintf(stderr, "ERROR: call to test_driver() failed.\n");
        goto err_pyprint;
    }
    
    Py_DECREF(pResult);
    Py_DECREF(pFunc);
    Py_DECREF(pMod);

    return(0);
    
err_pyprint:
    PyErr_Print();
err:
    return(1);
}

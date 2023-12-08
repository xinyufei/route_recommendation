#include <Python.h>
#include <iostream>
#include <string>
#include <map>

int main() {
    // Initialize the Python Interpreter
    Py_Initialize();

    // Import the Python module
    PyObject* pName = PyUnicode_DecodeFSDefault("algorithm_for_game");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    std::map<std::string, int> result;  // To store the result

    if (pModule != nullptr) {
        // Get the function from the module
        PyObject* pFunc = PyObject_GetAttrString(pModule, "get_recommended_route");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Create a Python dictionary
            PyObject* pDict = PyDict_New();
            PyDict_SetItem(pDict, PyUnicode_FromString("key1"), PyLong_FromLong(1));
            PyDict_SetItem(pDict, PyUnicode_FromString("key2"), PyLong_FromLong(2));

            // Create a Python list
            PyObject* pList = PyList_New(2);
            PyList_SetItem(pList, 0, PyLong_FromLong(1));
            PyList_SetItem(pList, 1, PyLong_FromLong(2));

            // Prepare the arguments for the function
            PyObject* pArgs = PyTuple_New(2); // Number of arguments
            PyTuple_SetItem(pArgs, 0, pDict);
            PyTuple_SetItem(pArgs, 1, pList);

            // Call the function
            PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != nullptr && PyDict_Check(pValue)) {
                PyObject *pKey, *pVal;
                Py_ssize_t pos = 0;

                // Iterate over the dictionary
                while (PyDict_Next(pValue, &pos, &pKey, &pVal)) {
                    if (PyUnicode_Check(pKey) && PyLong_Check(pVal)) {
                        std::string key = PyUnicode_AsUTF8(pKey);
                        int value = PyLong_AsLong(pVal);
                        result[key] = value;
                    }
                }
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
            }
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
    }

    // Finalize the Python Interpreter
    Py_Finalize();

    // Output the result
    for (const auto& kv : result) {
        std::cout << "Key: " << kv.first << ", Value: " << kv.second << std::endl;
    }

    return 0;
}

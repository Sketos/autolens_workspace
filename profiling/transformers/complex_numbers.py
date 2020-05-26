import os
import sys
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits


# ...
if os.environ["HOME"] == "/Users/ccbh87":
    COSMA_HOME = os.environ["COSMA_HOME_local"]
    COSMA_DATA = os.environ["COSMA7_DATA_local"]
elif os.environ["HOME"] == "/cosma/home/durham/dc-amvr1":
    COSMA_HOME = os.environ["COSMA_HOME_host"]
    COSMA_DATA = os.environ["COSMA7_DATA_host"]
else:
    raise ValueError

# ...
workspace_HOME_path = "{}/workspace".format(COSMA_HOME)
workspace_DATA_path = "{}/workspace".format(COSMA_DATA)
sys.path.append(workspace_HOME_path)
sys.path.append(workspace_DATA_path)

# ...
autolens_version = "0.45.0"

import autofit as af
af.conf.instance = af.conf.Config(
    config_path="{}/config_{}".format(
        workspace_DATA_path,
        autolens_version
    ),
    output_path="{}/output".format(
        workspace_DATA_path
    )
)
import autolens as al

from autoarray.util import transformer_util


if __name__ == "__main__":

    N = 100000
    array_1_real = np.random.normal(0.0, 1.0, size=N)
    array_1_imag = np.random.normal(0.0, 1.0, size=N)

    array_2_real = np.random.normal(0.0, 1.0, size=N)
    array_2_imag = np.random.normal(0.0, 1.0, size=N)

    start_outer_loop = time.time()
    for i in range(400):
        # # start = time.time()
        # array = np.multiply(
        #     array_1_real + 1j * array_1_imag,
        #     array_2_real + 1j * array_2_imag
        # )
        # end = time.time()
        # # print("t={}".format(end - start))

        #start = time.time()
        array_real, array_imag = transformer_util.multiply_complex_1d_arrays_jit(
            array_1_real=array_1_real,
            array_1_imag=array_1_imag,
            array_2_real=array_2_real,
            array_2_imag=array_2_imag
        )
        #end = time.time()
        #print("t={} (numba)".format(end - start))
    end_outer_loop = time.time()
    print("t={}".format(end_outer_loop - start_outer_loop))

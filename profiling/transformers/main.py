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

from autoarray.operators.transformer import TransformerNUFFT, TransformerFINUFFT


sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
from astropy_wrappers import Gaussian2D
import interferometry_utils.load_utils as interferometry_load_utils


class Image:
    def __init__(self, array_2d):
        self.array_2d = array_2d

    @property
    def in_1d_binned(self):
        return np.ndarray.flatten(self.array_2d)

    @property
    def in_2d_binned(self):
        return self.array_2d


def sparse_2d_array(shape_2d, size=5):

    array_2d = np.zeros(shape=shape_2d)

    idx_x = np.random.randint(shape_2d[0], size=size)
    idx_y = np.random.randint(shape_2d[1], size=size)
    array_2d[idx_x, idx_y] = 1

    return array_2d


pixel_scale = 0.05
n_pixels = 200


grid = al.Grid.uniform(
    shape_2d=(
        n_pixels,
        n_pixels
    ),
    pixel_scales=(
        pixel_scale,
        pixel_scale
    ),
    sub_size=1
)

tracer = al.Tracer.from_galaxies(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(
                    0.0,
                    0.0),
                axis_ratio=0.75,
                phi=45.0,
                einstein_radius=1.0
            ),
        ),
        al.Galaxy(
            redshift=2.0,
            pixelization=al.pix.VoronoiMagnification(
                shape=(20, 20)
            ),
            regularization=al.reg.Constant(
                coefficient=1.0
            ),
        )
    ]
)
mappers_of_planes = tracer.mappers_of_planes_from_grid(
    grid=grid,
    inversion_uses_border=False,
    preload_sparse_grids_of_planes=None
)

mapper = mappers_of_planes[-1]




def profilling_1(uv_wavelengths):


    for n_pixels, pixel_scale in zip([100, 200, 400], [0.05, 0.025, 0.0125]):
        print(n_pixels * pixel_scale)

        grid = al.Grid.uniform(
            shape_2d=(
                n_pixels,
                n_pixels
            ),
            pixel_scales=(
                pixel_scale,
                pixel_scale
            ),
            sub_size=1
        )

        array_2d = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=grid.shape_2d
        )

        transformer_NUFFT = TransformerNUFFT(
            uv_wavelengths=uv_wavelengths,
            grid=grid.in_radians
        )

        transformer_FINUFFT = TransformerFINUFFT(
            uv_wavelengths=uv_wavelengths,
            grid=grid.in_radians
        )

        NUFFT = []
        FINUFFT = []
        for i in range(10):

            start = time.time()
            visibilities_NUFFT = transformer_NUFFT.visibilities_from_image(
                image=Image(
                    array_2d=array_2d
                )
            )
            end = time.time()
            print("NUFFT:", end - start)
            NUFFT.append(end - start)

            start = time.time()
            visibilities_FINUFFT = transformer_FINUFFT.visibilities_from_image(
                image=Image(
                    array_2d=array_2d
                )
            )
            end = time.time()
            print("FINUFFT:", end - start)
            FINUFFT.append(end - start)



array_2d = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=grid.shape_2d
)

# array_2d = Gaussian2D(
#     shape_2d=grid.shape_2d,
#     amplitude=1.0,
#     x_mean=grid.shape_2d[0] / 2.0,
#     y_mean=grid.shape_2d[1] / 2.0,
#     x_stddev=grid.shape_2d[0] / 10.0,
#     y_stddev=grid.shape_2d[1] / 10.0,
#     theta=45.0
# )


if __name__ == "__main__":

    # NOTE: THE NUFFT transformers dont seem to be expoiting sparsity on the input matrix.

    filename = "/Users/ccbh87/Desktop/GitHub/UVgalpak3D/uv_wavelengths.fits"
    u_wavelengths, v_wavelengths = interferometry_load_utils.load_uv_wavelengths_from_fits(
        filename=filename
    )
    uv_wavelengths = np.stack(
        (u_wavelengths, v_wavelengths),
        axis=-1
    )

    uv_wavelengths_reshaped = uv_wavelengths.reshape(
        -1, uv_wavelengths.shape[-1]
    )

    uv_wavelengths_reshaped = np.concatenate(
        (
            uv_wavelengths_reshaped,
            uv_wavelengths_reshaped,
            uv_wavelengths_reshaped,
            uv_wavelengths_reshaped,
            uv_wavelengths_reshaped,
            uv_wavelengths_reshaped
        ),
        axis=0
    )
    #print(uv_wavelengths_reshaped.shape);exit()

    """
    transformer = TransformerNUFFT(
        uv_wavelengths=uv_wavelengths_reshaped,
        grid=grid.in_radians
    )

    array_2d = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=grid.shape_2d
    )

    array_2d_sparse = sparse_2d_array(
        shape_2d=grid.shape_2d,
        size=5
    )


    for i in range(10):
        start = time.time()
        visibilities = transformer.visibilities_from_image(
            image=Image(
                array_2d=array_2d
            )
        )
        end = time.time()
        print("dense:", end - start)

        start = time.time()
        visibilities_from_sparse = transformer.visibilities_from_image(
            image=Image(
                array_2d=array_2d_sparse
            )
        )
        end = time.time()
        print("sparse:", end - start)
    """

    # transformer_NUFFT = TransformerNUFFT(
    #     uv_wavelengths=uv_wavelengths_reshaped,
    #     grid=grid.in_radians
    # )

    transformer_FINUFFT = TransformerFINUFFT(
        uv_wavelengths=uv_wavelengths_reshaped,
        grid=grid.in_radians
    )

    # start = time.time()
    # visibilities_NUFFT = transformer_NUFFT.visibilities_from_image(
    #     image=Image(
    #         array_2d=array_2d
    #     )
    # )
    # end = time.time()
    # print("NUFFT:", end - start)
    #
    # start = time.time()
    # visibilities_FINUFFT = transformer_FINUFFT.visibilities_from_image(
    #     image=Image(
    #         array_2d=array_2d
    #     )
    # )
    # end = time.time()
    # print("FINUFFT:", end - start)
    #
    # dirty_image_NUFFT = transformer_NUFFT.image_from_visibilities(
    #     visibilities=visibilities_NUFFT
    # )
    #
    # dirty_image_FINUFFT = transformer_FINUFFT.image_from_visibilities(
    #     visibilities=visibilities_FINUFFT
    # )
    #
    #
    # figure, axes = plt.subplots(nrows=1, ncols=3)
    # axes[0].imshow(dirty_image_NUFFT)
    # axes[1].imshow(dirty_image_FINUFFT)
    # axes[2].imshow(dirty_image_NUFFT - dirty_image_FINUFFT)
    # plt.show()
    # exit()



    # for i in range(10):
    #     start = time.time()
    #     visibilities_NUFFT = transformer_NUFFT.visibilities_from_image(
    #         image=Image(
    #             array_2d=array_2d
    #         )
    #     )
    #     end = time.time()
    #     print("NUFFT:", end - start)
    #
    #     start = time.time()
    #     visibilities_FINUFFT = transformer_FINUFFT.visibilities_from_image(
    #         image=Image(
    #             array_2d=array_2d
    #         )
    #     )
    #     end = time.time()
    #     print("FINUFFT:", end - start)
    #
    #     # plt.figure()
    #     # plt.plot(
    #     #     visibilities_NUFFT[:, 0],
    #     #     visibilities_NUFFT[:, 1],
    #     #     linestyle="None",
    #     #     marker="o",
    #     #     markersize=10,
    #     #     color="b",
    #     #     alpha=0.5
    #     # )
    #     # plt.plot(
    #     #     visibilities_FINUFFT[:, 0],
    #     #     visibilities_FINUFFT[:, 1],
    #     #     linestyle="None",
    #     #     marker="o",
    #     #     markersize=5,
    #     #     color="r",
    #     #     alpha=1.0
    #     # )
    #     # plt.show()

    for i in range(10):
        print(i)

        # start = time.time()
        # transformer_NUFFT.transformed_mapping_matrices_from_mapping_matrix(
        #     mapping_matrix=mapper.mapping_matrix
        # )
        # end = time.time()
        # print("NUFFT:", end - start)

        start = time.time()
        transformer_FINUFFT.transformed_mapping_matrices_from_mapping_matrix(
            mapping_matrix=mapper.mapping_matrix
        )
        end = time.time()
        print("FINUFFT:", end - start)

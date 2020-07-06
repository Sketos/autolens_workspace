import os
import sys
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import astropy_wrappers as astropy_wrappers
import autolens_init_utils as autolens_init_utils


# NOTE:
workspace_paths = autolens_init_utils.get_workspace_paths(
    cosma_server="7"
)
# sys.path.append(workspace_paths["HOME"])
# sys.path.append(workspace_paths["DATA"])

# ...
#autolens_version = "0.45.0"
autolens_version = "0.46.2"

import autofit as af
af.conf.instance = af.conf.Config(
    config_path="{}/config_{}".format(
        workspace_paths["DATA"],
        autolens_version
    ),
    output_path="{}/output".format(
        workspace_paths["DATA"]
    )
)
import autolens as al


import autolens_utils.autolens_mapper_utils as autolens_mapper_utils
import interferometry_utils.load_utils as interferometry_load_utils


def load_uv_wavelengths_from_fits(filename):

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

    return uv_wavelengths_reshaped


pixel_scale = 0.05
n_pixels = 100

lens_redshift = 0.5
source_redshift = 2.0

filename = "{}/UVgalpak3D/uv_wavelengths.fits".format(
    os.environ["GitHub"]
)


# image = astropy_wrappers.Gaussian2D(
#     shape_2d=(n_pixels, n_pixels),
#     amplitude=1.0,
#     x_mean=n_pixels / 2.0,
#     y_mean=n_pixels / 2.0,
#     x_stddev=n_pixels / 20.0,
#     y_stddev=n_pixels / 40.0,
#     theta=45.0
# )
# reconstruction = np.ndarray.flatten(image)

if __name__ == "__main__":

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

    uv_wavelengths = load_uv_wavelengths_from_fits(
        filename=filename
    )

    lens_galaxy = al.Galaxy(
        redshift=lens_redshift,
        mass=al.mp.EllipticalIsothermal(
            centre=(
                0.0,
                0.0),
            axis_ratio=0.75,
            phi=45.0,
            einstein_radius=1.0
        ),
        shear=None
    )

    pixelization_shape_0 = 10
    pixelization_shape_1 = 10
    pixelization_shape_0 = n_pixels
    pixelization_shape_1 = n_pixels
    regularization_coefficient = 1.0

    pixelization_class = al.pix.Rectangular

    source_galaxy = al.Galaxy(
        redshift=source_redshift,
        pixelization=pixelization_class(
            shape=(pixelization_shape_0, pixelization_shape_1)
        ),
        regularization=al.reg.Constant(
            coefficient=regularization_coefficient
        ),
    )

    # tracer = al.Tracer.from_galaxies(
    #     galaxies=[
    #         lens_galaxy,
    #         al.Galaxy(
    #             redshift=source_redshift,
    #             light=al.lp.LightProfile()
    #         )
    #     ]
    # )
    tracer = al.Tracer.from_galaxies(
        galaxies=[
            lens_galaxy,
            source_galaxy
        ]
    )

    mappers_of_planes = tracer.mappers_of_planes_from_grid(
        grid=grid,
        inversion_uses_border=False,
        preload_sparse_grids_of_planes=None
    )
    mapper = mappers_of_planes[-1]

    # print(mapper.mapping_matrix.shape)
    # for i in range(mapper.mapping_matrix.shape[-1]):
    #
    #     mapping_i = mapper.mapping_matrix[:, i]
    #     mapping_i = mapping_i.reshape(grid.shape_2d)
    #     plt.figure()
    #     plt.imshow(mapping_i)
    #     plt.show()

    #reconstruction = np.random.normal(loc=0.0, scale=1.0, size=mapper.mapping_matrix.shape[-1])
    reconstruction = np.zeros(
        shape=mapper.mapping_matrix.shape[-1]
    )
    #N = 10
    #reconstruction[int(reconstruction.shape[0] / 2.0 - N):int(reconstruction.shape[0] / 2.0 + N)] = 1
    reconstruction[int(reconstruction.shape[0] / 2.0 + 50)] = 1

    mapped_reconstructed_image = autolens_mapper_utils.mapped_reconstruction(
        mapper=mapper, reconstruction=reconstruction
    )
    #print(mapped_reconstructed_image.in_2d.shape)
    print(np.sum(mapped_reconstructed_image))
    plt.figure()
    plt.imshow(mapped_reconstructed_image.in_2d)
    plt.show()

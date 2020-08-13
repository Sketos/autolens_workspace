import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

def paths(autolens_version):

    config_path = "{}/../configs/config_{}".format(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        autolens_version
    )
    if os.environ["HOME"].startswith("/cosma"):
        pass
    else:
        output_path = "{}/../outputs/output_{}".format(
            os.path.dirname(
                os.path.realpath(__file__)
            ),
            autolens_version
        )

    return config_path, output_path

autolens_version = "0.45.0"
config_path, output_path = paths(
    autolens_version=autolens_version
)

import autofit as af
af.conf.instance = af.conf.Config(
    config_path=config_path,
    output_path=output_path
)
import autoarray as aa
import autolens as al
import autolens.plot as aplt

from inversions import func_0

len_redshift = 0.5
src_redshift = 2.0

n_pixels = 100
pixel_scale = 0.05

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

real_space_mask = al.Mask.circular(
    shape_2d=grid.shape_2d,
    pixel_scales=grid.pixel_scales,
    sub_size=grid.sub_size,
    radius=2.0,
    centre=(0.0, 0.0)
)

transformer_class = al.TransformerFINUFFT

if __name__ == "__main__":

    uv_wavelengths = fits.getdata(
        filename="{}/uv_wavelengths.fits".format(
            os.path.dirname(
                os.path.realpath(__file__)
            )
        )
    )

    len_galaxy = al.Galaxy(
        redshift=len_redshift,
        mass=al.mp.EllipticalPowerLaw(
            centre=(-0.05, 0.05),
            axis_ratio=0.75,
            phi=45.0,
            einstein_radius=1.0,
            slope=2.0
        ),
    )

    src_galaxy = al.Galaxy(
        redshift=src_redshift,
        light=al.lp.EllipticalSersic(
            centre=(0.1, 0.0),
            axis_ratio=0.75,
            phi=45.0,
            intensity=0.001,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[
            len_galaxy,
            src_galaxy
        ]
    )

    transformer = transformer_class(
        uv_wavelengths=uv_wavelengths,
        grid=aa.structures.grids.MaskedGrid.from_mask(
            mask=real_space_mask
        ).in_radians
    )

    visibilities = tracer.profile_visibilities_from_grid_and_transformer(
        grid=grid,
        transformer=transformer
    )

    noise_map = np.random.normal(
        loc=0.0, scale=1.0 * 10**-2.0, size=visibilities.shape
    )

    interferometer = al.Interferometer(
        visibilities=al.Visibilities.manual_1d(
            np.add(visibilities, noise_map)
        ),
        noise_map=al.Visibilities.manual_1d(noise_map),
        uv_wavelengths=al.Visibilities.manual_1d(uv_wavelengths)
    )

    # plt.figure()
    # plt.plot(
    #     interferometer.visibilities[:, 0],
    #     interferometer.visibilities[:, 1],
    #     linestyle="None",
    #     marker="o",
    #     markersize=5,
    #     color="black"
    # )
    # plt.show()

    func_0(
        masked_interferometer=al.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=np.full(
                shape=interferometer.visibilities.shape,
                fill_value=False
            ),
            real_space_mask=real_space_mask,
            transformer_class=al.TransformerNUFFT
        ),
        tracer=al.Tracer.from_galaxies(
            galaxies=[
                len_galaxy,
                al.Galaxy(
                    redshift=src_redshift,
                    pixelization=al.pix.VoronoiMagnification(
                        shape=(20, 20)
                    ),
                    regularization=al.reg.Constant(
                        coefficient=1.0
                    )
                )
            ]
        ),
    )

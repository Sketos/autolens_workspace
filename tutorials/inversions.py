import autoarray as aa
import autolens as al

import matplotlib.pyplot as plt

def mapper_from_tracer_and_grid(tracer, grid):

    mappers_of_planes = tracer.mappers_of_planes_from_grid(
        grid=grid,
    )

    return mappers_of_planes[-1]

def func_0(masked_interferometer, tracer):

    inversion = aa.operators.inversion.inversions.InversionInterferometer.from_data_mapper_and_regularization(
        visibilities=masked_interferometer.visibilities,
        noise_map=masked_interferometer.noise_map,
        transformer=masked_interferometer.transformer,
        mapper=mapper_from_tracer_and_grid(
            tracer=tracer,
            grid=masked_interferometer.grid
        ),
        regularization=tracer.source_plane.regularization
    )


    # image = inversion.mapped_reconstructed_image
    #
    # #print(image.in_2d.shape)
    # plt.figure()
    # plt.imshow(image.in_2d)
    # plt.show()

    mapped_reconstructed_visibilities_from_mapped_reconstructed_image = masked_interferometer.transformer.visibilities_from_image(
        image=inversion.mapped_reconstructed_image
    )

    plt.figure()
    plt.plot(
        inversion.mapped_reconstructed_visibilities[:, 0],
        inversion.mapped_reconstructed_visibilities[:, 1],
        linestyle="None",
        marker="o",
        markersize=10,
        color="black",
        alpha=0.5
    )
    plt.plot(
        mapped_reconstructed_visibilities_from_mapped_reconstructed_image[:, 0],
        mapped_reconstructed_visibilities_from_mapped_reconstructed_image[:, 1],
        linestyle="None",
        marker="o",
        markersize=5,
        color="r"
    )
    plt.show()






# def func_1(uv_wavelengths, real_space_mask, tracer, transformer_class):
#
#     # transformers=[
#     #     transformer_class(
#     #         uv_wavelengths=masked_dataset.uv_wavelengths,
#     #         grid=multi_masked_dataset.grid.in_radians
#     #     )
#     #     for masked_dataset in multi_masked_dataset.masked_datasets
#     # ],
#
#     transformer = transformer_class(
#         uv_wavelengths=uv_wavelengths,
#         grid=aa.structures.grids.MaskedGrid.from_mask(
#             mask=mask
#         ).in_radians
#     )
#
#     # visibilities[i] = transformers[i].visibilities_from_image(
#     #     image=Image(
#     #         array_2d=lensed_cube[i]
#     #     )
#     # )
#
#     masked_interferometer = al.MaskedInterferometer(
#         interferometer=interferometer,
#         visibilities_mask=np.full(
#             shape=interferometer.visibilities.shape,
#             fill_value=False
#         ),
#         real_space_mask=real_space_mask,
#         transformer_class=transformer_class
#     )

if __name__ == "__main__":
    pass

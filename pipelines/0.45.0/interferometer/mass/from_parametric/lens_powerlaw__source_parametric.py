import autofit as af
import autolens as al


# def source_is_inversion_from_setup(setup):
#     if setup.source.type_tag in ["gaussian", "sersic", "ellipticalcoresersic"]:
#         return False
#     else:
#         return True
#
#
# def source_with_previous_model_or_instance(setup, source_redshift=None):
#     if setup.general.hyper_galaxies:
#
#         hyper_galaxy = af.PriorModel(al.HyperGalaxy)
#
#         hyper_galaxy.noise_factor = (
#             af.last.hyper_combined.model.galaxies.source.hyper_galaxy.noise_factor
#         )
#         hyper_galaxy.contribution_factor = (
#             af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.contribution_factor
#         )
#         hyper_galaxy.noise_power = (
#             af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.noise_power
#         )
#
#     else:
#
#         print("The is no hyper_galaxy")
#         hyper_galaxy = None
#
#     if setup.source.type_tag in ["gaussian", "sersic", "ellipticalcoresersic"]:
#         return al.GalaxyModel(
#             redshift=af.last.instance.galaxies.source.redshift,
#             light=af.last.model.galaxies.source.light,
#             hyper_galaxy=hyper_galaxy,
#         )
#
#     else:
#         # return al.GalaxyModel(
#         #     redshift=af.last.instance.galaxies.source.redshift,
#         #     pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
#         #     regularization=af.last.hyper_combined.instance.galaxies.source.regularization,
#         #     hyper_galaxy=hyper_galaxy,
#         # )
#
#         return al.GalaxyModel(
#             redshift=af.last.instance.galaxies.source.redshift,
#             pixelization=af.last.instance.galaxies.source.pixelization,
#             regularization=af.last.instance.galaxies.source.regularization,
#             hyper_galaxy=hyper_galaxy,
#         )
#
#         # return al.GalaxyModel(
#         #     redshift=af.last[index].instance.galaxies.source.redshift,
#         #     pixelization=af.last[index].instance.galaxies.source.pixelization,
#         #     regularization=af.last[index].instance.galaxies.source.regularization,
#         #     hyper_galaxy=hyper_galaxy,
#         # )


def update_phase_folders_from_setup(phase_folders, setup, source_type=None):

    for setup_type in [
        "general",
        "source",
        "mass"
    ]:
        if hasattr(setup, setup_type):
            obj = getattr(setup, setup_type)

            if setup_type == "general":
                phase_folders.append(
                    getattr(obj, "source_tag")
                )
            else:
                if obj.type_tag is None:
                    raise ValueError("...")
                phase_folders.append(
                    getattr(obj, "tag")
                )
        else:
            raise AttributeError("...")


def make_pipeline(
    setup,
    phase_folders,
    real_space_mask,
    lens_redshift,
    source_redshift,
    priors=None,
    pipeline_name="pipeline_mass__source_parametric",
    mass_type="PowerLaw",
    transformer_class=al.TransformerNUFFT,
    auto_positions_factor=None,
    positions_threshold=None,
    sub_size=1,
    pixel_scale_interpolation_grid=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
):

    phase_folders.append(pipeline_name)

    setup.set_mass_type(
        mass_type=mass_type
    )

    update_phase_folders_from_setup(
        phase_folders=phase_folders,
        setup=setup
    )

    # ... NOTE: Make it more elegant
    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    # ... NOTE : Make is more elegant?
    mass.centre = af.last.model.galaxies.lens.mass.centre
    mass.axis_ratio = af.last.model.galaxies.lens.mass.axis_ratio
    mass.phi = af.last.model.galaxies.lens.mass.phi
    mass.einstein_radius = af.last.model_absolute(
        a=0.3
    ).galaxies.lens.mass.einstein_radius

    # for attr, value in af.last.model.galaxies.lens.mass.__dict__.items():
    #     print(attr, value)

    if not setup.mass.no_shear:
        if af.last.model.galaxies.lens.shear is not None:
            shear = af.last.model.galaxies.lens.shear
        else:
            shear = al.mp.ExternalShear
    else:
        shear = None

    # ...
    lens = al.GalaxyModel(
        redshift=lens_redshift,
        mass=mass,
        shear=shear
    )

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1__lens_powerlaw{}__source".format(
            "_and_shear" if not setup.mass.no_shear else ''
        ),
        phase_folders=phase_folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=lens,
            source=af.last.model.galaxies.source,
        ),
        transformer_class=transformer_class,
        auto_positions_factor=auto_positions_factor,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        non_linear_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 100
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.evidence_tolerance = 0.8

    # phase1 = phase1.extend_with_multiple_hyper_phases(inversion=True)

    return al.PipelineDataset(
        pipeline_name,
        phase1
    )

import autofit as af
import autolens as al




def make_pipeline(
    setup,
    phase_folders,
    real_space_mask,
    lens_redshift,
    source_redshift,
    priors=None,
    pipeline_name="pipeline_mass__source_inversion"
    transformer_class=al.TransformerNUFFT,
    auto_positions_factor=None,
    positions_threshold=None,
    sub_size=1,
    pixel_scale_interpolation_grid=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
):

    setup.set_mass_type(mass_type="powerlaw")

    # ...
    if not setup.mass.no_shear:
        pipeline_name = "pipeline__lens_powerlaw_and_shear"
    else:
        pipeline_name = "pipeline__lens_powerlaw"
    pipeline_name += "__source"

    # # ...
    # if from_phase is None:
    #     pipeline_name += "__from__last_phase"
    # elif from_phase == "parametric":
    #     pipeline_name += "__from__parametric"
    # elif from_phase in [
    #     "inversion_phase_2",
    #     "inversion_phase_4"
    # ]:
    #     pipeline_name += "__from__" + from_phase
    # else:
    #     raise ValueError("Not a valid phase.")
    #phase_folders.append(pipeline_name + "__v" + al.__version__)
    phase_folders.append(pipeline_name)

    phase_folders.append(setup.general.tag)
    phase_folders.append(setup.source.tag)
    phase_folders.append(setup.mass.tag)

    # ... NOTE: Make it more elegant
    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    # ... NOTE : Make is more elegant?
    mass.centre = af.last.model.galaxies.lens.mass.centre
    mass.axis_ratio = af.last.model.galaxies.lens.mass.axis_ratio
    mass.phi = af.last.model.galaxies.lens.mass.phi
    mass.einstein_radius = af.last.model_absolute(
        a=0.3
    ).galaxies.lens.mass.einstein_radius

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

    # ...
    source = source_with_previous_model_or_instance(setup=setup, index=-1)

    # ...
    if not setup.mass.no_shear:
        phase1_name = "phase_1__lens_powerlaw_and_shear"
    else:
        phase1_name = "phase_1__lens_powerlaw"
    phase1_name += "__source"

    # # ...
    # if positions_threshold:
    #     phase1_name += "__with__positions"

    phase1 = al.PhaseInterferometer(
        phase_name=phase1_name,
        phase_folders=phase_folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=lens, source=source,
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

    # if not setup.general.hyper_fixed_after_source:
    #
    #     phase1 = phase1.extend_with_multiple_hyper_phases(inversion=True)

    return al.PipelineDataset(
        pipeline_name, phase1
    )

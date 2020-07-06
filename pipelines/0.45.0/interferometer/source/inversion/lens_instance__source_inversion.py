import autofit as af
import autolens as al


def make_pipeline(
    setup,
    phase_folders,
    real_space_mask,
    lens_redshift,
    source_redshift,
    instance,
    priors=None,
    pipeline_name="pipeline_source__inversion",
    transformer_class=al.TransformerNUFFT,
    auto_positions_factor=None,
    positions_threshold=None,
    sub_size=1,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
    evidence_tolerance=10.0,
):

    phase_folders.append(pipeline_name)

    for type in ["general", "source"]:
        if hasattr(setup, type):
            if type == "general":
                phase_folders.append(setup.general.source_tag)

            if type == "source":
                setup.set_source_type(
                    source_type=setup.source.inversion_tag
                )
                phase_folders.append(setup.source.tag)

    if "lens" in instance.keys():
        lens = instance["lens"]
    else:
        raise ValueError("...")

    source = al.GalaxyModel(
        redshift=source_redshift,
        pixelization=al.pix.VoronoiMagnification,
        regularization=al.reg.Constant,
    )

    source.pixelization.shape.shape_0 = af.UniformPrior(
        lower_limit=5,
        upper_limit=100
    )
    source.pixelization.shape.shape_1 = af.UniformPrior(
        lower_limit=5,
        upper_limit=100
    )
    source.regularization.coefficient = af.LogUniformPrior(
        lower_limit=10**-5.0,
        upper_limit=10**+5.0
    )

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1__lens_instance__source_inversion",
        phase_folders=phase_folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=lens,
            source=source,
        ),
        transformer_class=transformer_class,
        positions_threshold=positions_threshold,
        auto_positions_factor=auto_positions_factor,
        sub_size=sub_size,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        non_linear_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8
    phase1.optimizer.evidence_tolerance = 0.1

    return al.PipelineDataset(
        pipeline_name,
        phase1
    )

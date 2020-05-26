import autofit as af
import autolens as al


def set_prior_init(type, **kwargs):

    prior = getattr(
        af, "{}Prior".format(type)
    )

    if type in ["Gaussian"]:
        if not all(
            key in kwargs
            for key in ("mean", "sigma")
        ):
            raise ValueError(
                "{} and {}".format("mean", "sigma")
            )
    elif type in ["Uniform", "LogUniform"]:
        if not all(
            key in kwargs
            for key in ("lower_limit", "upper_limit")
        ):
            raise ValueError(
                "{} and {}".format("lower_limit", "upper_limit")
            )
    else:
        raise ValueError(
            "{} is not supported".format(type)
        )

    return prior(**kwargs)


def set_priors_helper(dict):

    if "type" in dict.keys():
        type = dict["type"]
    else:
        raise ValueError("...")

    kwargs = {
        x: dict[x] for x in dict
        if x not in ["type"]
    }

    return set_prior_init(
        type, **kwargs
    )


# NOTE: Make it so that the GalaxyModel can be recognised. Atm it does not recognise if
# it is applied to the lens or a subhalo
# TODO: Take as input a list of GalaxyModels (e.g. for lens, subhalo, source)
def set_priors(GalaxyModel, priors=None):

    if not isinstance(GalaxyModel, al.GalaxyModel):
        raise ValueError("...")

    if priors is not None: # TODO: Check is priors is a dictionary

        for i_key in priors.keys():

            if hasattr(GalaxyModel, i_key):

                if isinstance(priors[i_key], dict):

                    for j_key, priors_dict in priors[i_key].items():

                        if hasattr(
                            getattr(GalaxyModel, i_key),
                            j_key
                        ):

                            setattr(
                                getattr(GalaxyModel, i_key),
                                j_key,
                                set_priors_helper(
                                    dict=priors_dict
                                )
                            )


def update_phase_folders_from_setup(phase_folders, setup, source_type=None):

    for setup_type in ["general", "source"]:
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
    pipeline_name="pipeline_source__parametric",
    source_type="EllipticalSersic",
    transformer_class=al.TransformerNUFFT,
    auto_positions_factor=None,
    positions_threshold=None,
    sub_size=2,
    evidence_tolerance=10.0,
):

    phase_folders.append(pipeline_name)

    setup.set_source_type(
        source_type=source_type
    )

    update_phase_folders_from_setup(
        phase_folders=phase_folders,
        setup=setup
    )

    if not setup.source.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    lens = al.GalaxyModel(
        redshift=lens_redshift,
        mass=al.mp.EllipticalIsothermal,
        shear=shear
    )

    if hasattr(al.lp, source_type):
        source = al.GalaxyModel(
            redshift=source_redshift,
            light=getattr(al.lp, source_type)
        )
    else:
        raise AttributeError(
            "{} is not supported".format(source_type)
        )

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1__lens_sie{}__source_{}".format(
            "_and_shear" if not setup.source.no_shear else '',
            source_type
        ),
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
        non_linear_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 100
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.evidence_tolerance = evidence_tolerance

    return al.PipelineDataset(
        pipeline_name,
        phase1
    )

import autofit as af
import autolens as al


# def set_prior_init(type, **kwargs):
#
#     prior = getattr(
#         af, "{}Prior".format(type)
#     )
#
#     if type in ["Gaussian"]:
#         if not all(
#             key in kwargs
#             for key in ("mean", "sigma")
#         ):
#             raise ValueError(
#                 "{} and {}".format("mean", "sigma")
#             )
#     elif type in ["Uniform", "LogUniform"]:
#         if not all(
#             key in kwargs
#             for key in ("lower_limit", "upper_limit")
#         ):
#             raise ValueError(
#                 "{} and {}".format("lower_limit", "upper_limit")
#             )
#     else:
#         raise ValueError(
#             "{} is not supported".format(type)
#         )
#
#     return prior(**kwargs)
#
#
# def set_priors_helper(dict):
#
#     if "type" in dict.keys():
#         type = dict["type"]
#     else:
#         raise ValueError("...")
#
#     kwargs = {
#         x: dict[x] for x in dict
#         if x not in ["type"]
#     }
#
#     return set_prior_init(
#         type, **kwargs
#     )
#
#
# # NOTE: Make it so that the GalaxyModel can recognise if the mass profile belongs to subhalo or a SIE. Atm it does not recognise if
# # it is applied to the lens or a subhalo
# def set_priors(GalaxyModels, priors=None):
#
#     if isinstance(GalaxyModels, list):
#         for GalaxyModel in GalaxyModels:
#             if not isinstance(GalaxyModel, al.GalaxyModel):
#                 raise ValueError("...")
#     else:
#         raise ValueError(
#             "must be a list."
#         )
#
#     if priors is not None: # TODO: Check is priors is a dictionary
#
#         for i_key in priors.keys():
#
#             for GalaxyModel in GalaxyModels:
#
#                 if hasattr(GalaxyModel, i_key):
#
#                     if isinstance(priors[i_key], dict):
#
#                         for j_key, priors_dict in priors[i_key].items():
#
#                             if hasattr(
#                                 getattr(GalaxyModel, i_key),
#                                 j_key
#                             ):
#
#                                 setattr(
#                                     getattr(GalaxyModel, i_key),
#                                     j_key,
#                                     set_priors_helper(
#                                         dict=priors_dict
#                                     )
#                                 )

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
    redshift=1.0,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    evidence_tolerance=100.0,
):

    phase_folders.append(pipeline_name)

    setup.set_source_type(
        source_type=source_type
    )

    update_phase_folders_from_setup(
        phase_folders=phase_folders,
        setup=setup
    )

    sersic_0 = af.PriorModel(al.lp.EllipticalSersic)

    phase1 = al.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            galaxy=al.GalaxyModel(
                redshift=redshift,
                sersic_0=sersic_0,
            ),
        ),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 100
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.evidence_tolerance = evidence_tolerance


    sersic_1 = af.PriorModel(al.lp.EllipticalSersic)
    #sersic_1.centre_0 = phase1.result.model.galaxies.galaxy.sersic_0.centre_0
    #sersic_1.centre_1 = phase1.result.model.galaxies.galaxy.sersic_0.centre_1

    phase2 = al.PhaseImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            galaxy=al.GalaxyModel(
                redshift=redshift,
                sersic_0=phase1.result.model.galaxies.galaxy.sersic_0,
                sersic_1=sersic_1,
            ),
        ),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 100
    phase2.optimizer.sampling_efficiency = 0.2
    phase2.optimizer.evidence_tolerance = evidence_tolerance


    sersic_2 = af.PriorModel(al.lp.EllipticalSersic)
    #sersic_2.centre_0 = phase2.result.model.galaxies.galaxy.sersic_0.centre_0
    #sersic_2.centre_1 = phase2.result.model.galaxies.galaxy.sersic_0.centre_1

    phase3 = al.PhaseImaging(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            galaxy=al.GalaxyModel(
                redshift=redshift,
                sersic_0=phase2.result.model.galaxies.galaxy.sersic_0,
                sersic_1=phase2.result.model.galaxies.galaxy.sersic_1,
                sersic_2=sersic_2,
            ),
        ),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 100
    phase3.optimizer.sampling_efficiency = 0.2
    phase3.optimizer.evidence_tolerance = evidence_tolerance

    return al.PipelineDataset(
        pipeline_name,
        phase1,
        phase2,
        phase3
    )

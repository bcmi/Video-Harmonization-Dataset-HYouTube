from iharm.model.base import SSAMImageHarmonization


BMCONFIGS = {
    'ssam256': {
        'model': SSAMImageHarmonization,
        'params': {'depth': 4, 'batchnorm_from': 2, 'attend_from': 2}
    },
    'improved_ssam256': {
        'model': SSAMImageHarmonization,
        'params': {'depth': 4, 'ch': 32, 'image_fusion': True, 'attention_mid_k': 0.5,
                   'batchnorm_from': 2, 'attend_from': 2}
    },
    'improved_ssam512': {
        'model': SSAMImageHarmonization,
        'params': {'depth': 6, 'ch': 32, 'image_fusion': True, 'attention_mid_k': 0.5,
                   'batchnorm_from': 2, 'attend_from': 3}
    },

}

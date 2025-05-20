import logging

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from saicinpainting.training.modules.pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, \
    NLayerDiscriminator, MultidilatedNLayerDiscriminator

def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')

    if kind == 'pix2pixhd_multidilated':
        return MultiDilatedGlobalGenerator(**kwargs)
    
    if kind == 'pix2pixhd_global':
        return GlobalGenerator(**kwargs)

    if kind == 'ffc_resnet':
        freeze_color_layers = kwargs.pop('freeze_color_layers', getattr(config, 'freeze_color_layers', False))
        num_frozen_layers = kwargs.pop('num_frozen_layers', getattr(config, 'num_frozen_layers', 2))
        return FFCResNetGenerator(
            **kwargs,
            freeze_color_layers=freeze_color_layers,
            num_frozen_layers=num_frozen_layers
        )

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')

    if kind == 'pix2pixhd_nlayer_multidilated':
        return MultidilatedNLayerDiscriminator(**kwargs)

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)

    raise ValueError(f'Unknown discriminator kind {kind}')

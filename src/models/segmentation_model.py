import segmentation_models_pytorch as smp


def _resolve_in_channels(config):
    """Decide how many input channels the model should accept.

    If CS 136 preprocessing is on AND the Canny-edge channel is enabled,
    the dataset stacks edges as a 4th channel, so the model must accept 4.
    Otherwise stick with 3.
    """
    cs136 = config.get('cs136_preprocessing', {}) or {}
    if cs136.get('enabled') and cs136.get('canny_channel', {}).get('enabled', True):
        return 4
    return 3


def create_model(config):
    """
    Creates the segmentation model using segmentation_models_pytorch.
    Supported architectures: DeepLabV3Plus, Unet, FPN, etc.
    """
    arch = config['model'].get('architecture', 'DeepLabV3Plus')
    encoder = config['model'].get('encoder', 'resnet50')
    weights = config['model'].get('encoder_weights', 'imagenet')
    num_classes = config['dataset'].get('num_classes', 19)
    in_channels = _resolve_in_channels(config)

    # We want None here instead of "imagenet" if it's set to "None" string or null
    if weights == "None" or weights == "none" or weights is None:
        weights = None

    print(f"Loading {arch} with encoder {encoder}, weights={weights}, in_channels={in_channels}")

    if arch == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif arch == "Unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Architecture {arch} is not supported.")

    return model

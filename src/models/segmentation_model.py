import segmentation_models_pytorch as smp

def create_model(config):
    """
    Creates the segmentation model using segmentation_models_pytorch.
    Supported architectures: DeepLabV3Plus, Unet, FPN, etc.
    """
    arch = config['model'].get('architecture', 'DeepLabV3Plus')
    encoder = config['model'].get('encoder', 'resnet50')
    weights = config['model'].get('encoder_weights', 'imagenet')
    num_classes = config['dataset'].get('num_classes', 19)
    
    # We want None here instead of "imagenet" if it's set to "None" string or null
    if weights == "None" or weights == "none" or weights is None:
        weights = None

    print(f"Loading {arch} with encoder {encoder} and weights {weights}")

    if arch == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
    elif arch == "Unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Architecture {arch} is not supported.")

    return model

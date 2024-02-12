from typing import List


from torchvision.transforms import v2


def _get_color_distortion(s=1.0) -> v2.Transform:
    color_jitter = v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = v2.RandomApply([color_jitter], p=0.8)

    rnd_gray = v2.RandomGrayscale(p=0.2)

    color_distort = v2.Compose([
        rnd_color_jitter,
        rnd_gray])

    return color_distort


def _get_gaussian_blur() -> v2.Transform:
    gaussian_blur = v2.GaussianBlur(22, (0.1, 2.0))

    rnd_gaussian_blur = v2.RandomApply(gaussian_blur, p=0.5)

    return rnd_gaussian_blur


def _get_random_resized_crop() -> v2.Transform:
    return v2.RandomResizedCrop(224)


def _get_random_horizontal_flip() -> v2.Transform:
    return v2.RandomHorizontalFlip(p=0.5)


def get_augmentations(augs: List[str]) -> v2.Transform:
    transforms = [_get_random_resized_crop()]

    aug_map = {
        'colordistortion': _get_color_distortion,
        'gaussianblur': _get_gaussian_blur,
        'horizontalflip': _get_random_horizontal_flip
    }

    supported_augs = sorted(aug_map.keys())
    supported_augs = ', '.join(supported_augs)

    for aug in augs:
        assert aug.lower(
        ) in supported_augs, f'Unexpected augmentation: {aug}, supported augmentations: {supported_augs}'

        aug_fn = aug_map[aug.lower()]

        transforms.append(aug_fn())

    return v2.Compose(transforms)

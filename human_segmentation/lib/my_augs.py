from albumentations import (
    Compose, HorizontalFlip, VerticalFlip,  CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, OneOf , RandomSizedCrop, PadIfNeeded, RandomRotate90, ElasticTransform,
    GridDistortion, OpticalDistortion, GaussNoise, Blur
)

def get_augs():
    light_augs = Compose([    
                        HorizontalFlip(p=1),              
                        Blur(blur_limit=3, p=1),
                        RandomContrast(limit=0.3, p=0.5),
                        RandomGamma(gamma_limit=(50, 100), p=0.5),
                        RandomBrightness(limit=0.5, p=0.5),
                        HueSaturationValue(hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=5, p=.9),
                        CLAHE(p=1.0, clip_limit=2.0)])
    heavy_augs = Compose([
                        OneOf([
                            ElasticTransform(p=0.5, alpha=40, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            GridDistortion(p=0.5),
                            OpticalDistortion(p=0.5, distort_limit=0.2, shift_limit=0.5)                  
                            ], p=0.8),
                        OneOf([RandomSizedCrop(min_max_height=(200, 240), height=256, width=256, p=0.5),
                            PadIfNeeded(min_height=256, min_width=256, p=0.5)], p=1),    
                        HorizontalFlip(p=1),              
                        Blur(blur_limit=3, p=1),
                        RandomContrast(limit=0.3, p=0.5),
                        RandomGamma(gamma_limit=(50, 100), p=0.5),
                        RandomBrightness(limit=0.5, p=0.5),
                        HueSaturationValue(hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=5, p=.9),
                        CLAHE(p=1.0, clip_limit=2.0)])

    psn_augs = Compose([
                        OneOf([
                            ElasticTransform(p=0.5, alpha=40, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            GridDistortion(p=0.5),
                            OpticalDistortion(p=0.5, distort_limit=0.2, shift_limit=0.5)                  
                            ], p=0.8),
                        OneOf([RandomSizedCrop(min_max_height=(200, 240), height=240, width=240, p=0.5),
                            PadIfNeeded(min_height=240, min_width=240, p=0.5)], p=1),    
                        HorizontalFlip(p=1),              
                        Blur(blur_limit=3, p=1),
                        RandomContrast(limit=0.3, p=0.5),
                        RandomGamma(gamma_limit=(50, 100), p=0.5),
                        RandomBrightness(limit=0.5, p=0.5),
                        HueSaturationValue(hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=5, p=.9),
                        CLAHE(p=1.0, clip_limit=2.0)])
    return light_augs, heavy_augs, psn_augs
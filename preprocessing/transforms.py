import cv2
import numpy as np
from skimage.filters import threshold_sauvola

def to_grayscale(image:np.ndarray)->np.ndarray:
    if len(image.shape)==2:
        return image
    if image.shape[2]==1:
        return image.squeeze(axis=2)
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def binarize_otsu(image:np.ndarray)->np.ndarray:
    _,binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def binarize_sauvola(image:np.ndarray,window_size:int =25,k:float=0.2)->np.ndarray:
    threshold_map = threshold_sauvola(image,window_size=window_size,k=k)
    binary = np.zeros_like(image)
    binary[image<threshold_map] = 255
    return binary

def binarize(method:str,image:np.ndarray,**kwargs)->np.ndarray:
    if method=="otsu":
        return binarize_otsu(image)
    elif method=="sauvola":
        return binarize_sauvola(image,**kwargs)
    else:
        raise ValueError(f"Unknown binarization method {method}")


def denoise_morphological(image:np.ndarray,kernel_size:int=3)->np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    cleaned = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    return cleaned

def denoise_median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.medianBlur(image,kernel_size)

def denoise_gaussian(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)

def denoise(image: np.ndarray, method: str = "morphological", **kwargs) -> np.ndarray:
    methods = {
        "morphological": denoise_morphological,
        "median": denoise_median,
        "gaussian": denoise_gaussian,
    }
    if method not in methods:
        raise ValueError(f"Unknown denoising method: {method}")
    return methods[method](image, **kwargs)

def resize_and_pad(image:np.ndarray,target_height:int=64,target_width:int=2048,pad_alignment:str="left"):
    h,w = image.shape[:2]
    scale = target_height / h
    new_width = max(1, int(round(w * scale)))

    resized = cv2.resize(image,(new_width,target_height),interpolation=cv2.INTER_AREA)

    if new_width < target_width:
        pad_width = target_width - new_width
        pad_alignment = (pad_alignment or "left").lower()
        if pad_alignment == "center":
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
        elif pad_alignment == "right":
            left_pad = 0
            right_pad = pad_width
        else:
            left_pad = pad_width
            right_pad = 0
        resized = cv2.copyMakeBorder(
                resized,
                top=0, bottom=0,
                left=left_pad, right=right_pad,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

    return resized

def normalize(image:np.ndarray)->np.ndarray:
    return image.astype(np.float32)/255.0


def _elastic_distortion(image: np.ndarray, alpha: float = 36, sigma: float = 6) -> np.ndarray:
    shape = image.shape
    random_state = np.random.RandomState(None)

    dx = (random_state.rand(*shape) * 2 - 1)
    dy = (random_state.rand(*shape) * 2 - 1)

    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return distorted


def apply_augmentations(image: np.ndarray, cfg: dict) -> np.ndarray:
    aug = cfg or {}
    img = image.copy()

    # Affine: rotation + scale + translation
    if aug.get("affine", {}).get("enabled", False):
        rot = float(aug.get("rotation_range", 0))
        angle = np.random.uniform(-rot, rot)
        scale = np.random.uniform(1 - aug.get("scale", 0.05), 1 + aug.get("scale", 0.05))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        tx = int(np.random.uniform(-aug.get("translate", 0) * w, aug.get("translate", 0) * w))
        ty = int(np.random.uniform(-aug.get("translate", 0) * h, aug.get("translate", 0) * h))
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Blur
    if aug.get("blur", {}).get("enabled", False):
        k = aug.get("blur", {}).get("kernel_size", 3)
        if k > 0:
            img = cv2.GaussianBlur(img, (k, k), 0)

    # Brightness / contrast
    if aug.get("brightness_contrast", {}).get("enabled", False):
        b_range = aug.get("brightness_contrast", {}).get("brightness_range", 0.15)
        c_range = aug.get("brightness_contrast", {}).get("contrast_range", 0.15)
        alpha = 1.0 + np.random.uniform(-c_range, c_range)
        beta = int(np.random.uniform(-b_range * 255, b_range * 255))
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Elastic
    if aug.get("elastic_distortion", {}).get("enabled", False):
        alpha = aug.get("elastic_distortion", {}).get("alpha", 36)
        sigma = aug.get("elastic_distortion", {}).get("sigma", 6)
        img = _elastic_distortion(img, alpha=alpha, sigma=sigma)

    return img


class ManuscriptPreprocessor:

    def __init__(self, config: dict):
      
        self.bin_method = config.get("binarization", {}).get("method", "sauvola")
        self.bin_window = config.get("binarization", {}).get("window_size", 25)
        self.bin_k = config.get("binarization", {}).get("k", 0.2)

        self.denoise_enabled = config.get("denoising", {}).get("enabled", True)
        self.denoise_method = config.get("denoising", {}).get("method", "morphological")
        self.denoise_kernel = config.get("denoising", {}).get("kernel_size", 3)
        self.pad_alignment = config.get("pad_alignment", "left")

    def __call__(self, image: np.ndarray, target_height: int = 64, target_width: int = 2048) -> np.ndarray:
        img = to_grayscale(image)
        # binarize expects image first, then method, then kwargs
        img = binarize(self.bin_method, img, window_size=self.bin_window, k=self.bin_k)

        if self.denoise_enabled:
            img = denoise(img, method=self.denoise_method, kernel_size=self.denoise_kernel)

        img = resize_and_pad(
            img,
            target_height=target_height,
            target_width=target_width,
            pad_alignment=self.pad_alignment,
        )
        img = normalize(img)
        return img

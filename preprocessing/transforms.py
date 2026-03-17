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

def resize_and_pad(image:np.ndarray,target_height:int=64,target_width:int=384):
    h,w = image.shape[:2]
    scale = target_height / h
    new_width = int(w* scale)
    if new_width > target_width:
        resized = cv2.resize(image,(target_width,target_height),interpolation=cv2.INTER_AREA)
    else :
        resized = cv2.resize(image,(new_width,target_height),interpolation=cv2.INTER_AREA)

        pad_width = target_width - new_width
        resized = cv2.copyMakeBorder(
                resized,
                top=0, bottom=0,
                left=pad_width, right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
    return resized

def normalize(image:np.ndarray)->np.ndarray:
    return image.astype(np.float32)/255.0


class ManuscriptPreprocessor:

    def __init__(self, config: dict):
      
        self.bin_method = config.get("binarization", {}).get("method", "sauvola")
        self.bin_window = config.get("binarization", {}).get("window_size", 25)
        self.bin_k = config.get("binarization", {}).get("k", 0.2)

        self.denoise_enabled = config.get("denoising", {}).get("enabled", True)
        self.denoise_method = config.get("denoising", {}).get("method", "morphological")
        self.denoise_kernel = config.get("denoising", {}).get("kernel_size", 3)

    def __call__(self, image: np.ndarray, target_height: int = 64, target_width: int = 384) -> np.ndarray:
        img = to_grayscale(image)
        # binarize expects image first, then method, then kwargs
        img = binarize(self.bin_method, img, window_size=self.bin_window, k=self.bin_k)

        if self.denoise_enabled:
            img = denoise(img, method=self.denoise_method, kernel_size=self.denoise_kernel)

        img = resize_and_pad(img, target_height=target_height, target_width=target_width)
        img = normalize(img)
        return img

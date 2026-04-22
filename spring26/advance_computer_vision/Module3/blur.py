import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, ifftshift

def create_gaussian_kernel(size=15, sigma=3.0):
    """Create normalized Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def spatial_blur(image, kernel):
    """Spatial convolution with periodic boundaries"""
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_WRAP)

def fourier_blur(image, kernel):
    """Blur via FFT multiplication with proper kernel centering."""
    # 1. Pad kernel to image size, center it
    kernel_padded = np.zeros_like(image, dtype=np.float64)
    ky, kx = kernel.shape
    iy, ix = image.shape
    sy = iy // 2 - ky // 2
    sx = ix // 2 - kx // 2
    kernel_padded[sy:sy+ky, sx:sx+kx] = kernel
    
    # 2. Shift kernel for FFT
    kernel_shifted = ifftshift(kernel_padded)
    
    # 3. FFT multiplication
    F_img = fft2(image.astype(np.float64))
    F_kernel = fft2(kernel_shifted)
    F_blurred = F_img * F_kernel
    blurred = np.real(ifft2(F_blurred))
    return blurred

def main():
    # Load ANY image
    img = cv2.imread('images/strawberry.jpg', 0)  # grayscale
    img = img.astype(np.float64) / 255.0
    
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(15, 3.0)
    
    print("=== Testing with Image ===")
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    # Method 1: Spatial with periodic boundaries (should match FFT)
    blurred_spatial = spatial_blur(img, kernel)

    # Method 2: Fourier domain
    blurred_fourier = fourier_blur(img, kernel)
    
    # Compute MSEs
    mse_periodic_vs_fourier = np.mean((blurred_spatial - blurred_fourier) ** 2)
    print(f"MSE (Spatial vs Fourier): {mse_periodic_vs_fourier:.2e}")

    # Compute max difference
    max_diff = np.max(np.abs(blurred_spatial - blurred_fourier))
    print(f"Max difference (Spatial vs Fourier): {max_diff:.2e}")
    
    # Visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    # Row 1: Images
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')

    axes[1].imshow(blurred_spatial, cmap='gray')
    axes[1].set_title('Spatial Blur')

    axes[2].imshow(blurred_fourier, cmap='gray')
    axes[2].set_title('Fourier Blur')

    plt.suptitle('Convolution Theorem: Spatial vs Fourier Domain Blurring', fontsize=14)
    plt.tight_layout()
    plt.savefig('real_image_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
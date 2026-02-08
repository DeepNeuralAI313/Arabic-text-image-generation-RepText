"""
Script to download popular Arabic fonts for RepText training.
Downloads free Arabic fonts from Google Fonts.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_google_font(font_name: str, output_dir: str):
    """
    Download font from Google Fonts.
    Note: This uses a simplified approach. For production, consider using the Google Fonts API.
    """
    # Direct download links for popular Arabic fonts
    font_urls = {
        "Amiri": "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf",
        "Cairo": "https://github.com/google/fonts/raw/main/ofl/cairo/Cairo-Regular.ttf",
        "Tajawal": "https://github.com/google/fonts/raw/main/ofl/tajawal/Tajawal-Regular.ttf",
        "Almarai": "https://github.com/google/fonts/raw/main/ofl/almarai/Almarai-Regular.ttf",
        "Changa": "https://github.com/google/fonts/raw/main/ofl/changa/Changa-Regular.ttf",
        "Lalezar": "https://github.com/google/fonts/raw/main/ofl/lalezar/Lalezar-Regular.ttf",
        "Markazi": "https://github.com/google/fonts/raw/main/ofl/markazitext/MarkaziText-Regular.ttf",
        "Reem Kufi": "https://github.com/google/fonts/raw/main/ofl/reemkufi/ReemKufi-Regular.ttf",
        "Scheherazade": "https://github.com/google/fonts/raw/main/ofl/scheherazadenew/ScheherazadeNew-Regular.ttf",
        "Harmattan": "https://github.com/google/fonts/raw/main/ofl/harmattan/Harmattan-Regular.ttf",
    }
    
    if font_name not in font_urls:
        print(f"Font {font_name} not found in available fonts.")
        return False
    
    url = font_urls[font_name]
    filename = url.split('/')[-1]
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        print(f"{filename} already exists, skipping...")
        return True
    
    try:
        print(f"Downloading {font_name}...")
        download_url(url, output_path)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {font_name}: {e}")
        return False


def main():
    """Download recommended Arabic fonts."""
    
    # Create output directory
    output_dir = "arabic_fonts"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Arabic Fonts Downloader for RepText")
    print("=" * 50)
    print()
    
    # List of recommended fonts
    recommended_fonts = [
        "Amiri",
        "Cairo", 
        "Tajawal",
        "Almarai",
        "Changa",
        "Lalezar",
        "Markazi",
        "Reem Kufi",
        "Scheherazade",
        "Harmattan"
    ]
    
    print(f"Downloading {len(recommended_fonts)} Arabic fonts to '{output_dir}/'...")
    print()
    
    success_count = 0
    for font_name in recommended_fonts:
        if download_google_font(font_name, output_dir):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"Download complete!")
    print(f"Successfully downloaded {success_count}/{len(recommended_fonts)} fonts")
    print(f"Fonts saved to: {os.path.abspath(output_dir)}")
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. Verify fonts by checking the arabic_fonts/ directory")
    print("2. Create arabic_texts.txt with your Arabic text samples")
    print("3. Run: python prepare_arabic_dataset.py --output_dir ./arabic_training_data --font_dir ./arabic_fonts --num_samples 10000")
    print("4. Run: accelerate launch train_arabic.py --config train_config.yaml")


if __name__ == '__main__':
    main()

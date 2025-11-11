#!/usr/bin/env python3
"""
Download Alternative Datasets for AI-Generated Image Detection
Pilihan dataset yang lebih kecil dan mudah digunakan
"""

import os
import sys
import subprocess
from pathlib import Path

# Dataset options with their details
DATASETS = {
    "1": {
        "name": "CIFAKE",
        "size": "~25GB",
        "description": "120K real + 120K fake (CIFAR-10 style)",
        "url": "kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images",
        "type": "kaggle",
        "recommended": True
    },
    "2": {
        "name": "GenImage",
        "size": "~10GB", 
        "description": "Multiple GAN/Diffusion generators",
        "url": "kaggle datasets download -d khaledzsa/genimage",
        "type": "kaggle",
        "recommended": True
    },
    "3": {
        "name": "DiffusionDB",
        "size": "~5GB (subset)",
        "description": "Stable Diffusion generated images",
        "url": "huggingface-cli download --repo-type dataset poloclub/diffusiondb --include '2m_first_1k.zip'",
        "type": "huggingface",
        "recommended": False
    },
    "4": {
        "name": "RAISE",
        "size": "~4GB",
        "description": "Real images only (baseline)",
        "url": "http://loki.disi.unitn.it/RAISE/",
        "type": "manual",
        "recommended": False
    }
}

def check_kaggle_setup():
    """Check if Kaggle CLI is installed and configured"""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        return False
    return False

def setup_kaggle():
    """Install and setup Kaggle CLI"""
    print("\n" + "="*60)
    print("🔧 SETUP KAGGLE CLI")
    print("="*60)
    
    # Install kaggle
    print("\n📦 Installing kaggle package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    
    print("\n✅ Kaggle installed!")
    print("\n" + "="*60)
    print("📋 LANGKAH SETUP KAGGLE API:")
    print("="*60)
    print("\n1. Login ke Kaggle: https://www.kaggle.com/")
    print("2. Go to Account: https://www.kaggle.com/settings")
    print("3. Scroll ke 'API' section")
    print("4. Click 'Create New Token'")
    print("5. Download kaggle.json")
    print("6. Pindahkan kaggle.json ke ~/.kaggle/")
    print("\nAtau jalankan:")
    print("  mkdir -p ~/.kaggle")
    print("  mv ~/Downloads/kaggle.json ~/.kaggle/")
    print("  chmod 600 ~/.kaggle/kaggle.json")
    print("\n" + "="*60)
    
    input("\n⏸️  Tekan ENTER setelah setup kaggle.json...")
    
def check_huggingface_setup():
    """Check if HuggingFace CLI is ready"""
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def setup_huggingface():
    """Install and setup HuggingFace CLI"""
    print("\n" + "="*60)
    print("🔧 SETUP HUGGINGFACE CLI")
    print("="*60)
    
    # Install huggingface-hub
    print("\n📦 Installing huggingface-hub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], check=True)
    
    print("\n✅ HuggingFace Hub installed!")
    print("\n" + "="*60)
    print("📋 LANGKAH LOGIN:")
    print("="*60)
    print("\n1. Buat token: https://huggingface.co/settings/tokens")
    print("2. Pilih type: 'Read'")
    print("3. Copy token")
    print("4. Login dengan: huggingface-cli login")
    print("\n" + "="*60)
    
    subprocess.run(["huggingface-cli", "login"])

def download_cifake(output_dir):
    """Download CIFAKE dataset"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING CIFAKE DATASET")
    print("="*60)
    print("\nSize: ~25GB")
    print("Content: 120K real + 120K AI-generated images")
    print("Format: CIFAR-10 style (32x32)")
    print("\n" + "="*60)
    
    if not check_kaggle_setup():
        setup_kaggle()
    
    # Create output directory
    cifake_dir = Path(output_dir) / "cifake"
    cifake_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    print(f"\n📂 Downloading to: {cifake_dir}")
    cmd = f"cd {cifake_dir} && kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images"
    subprocess.run(cmd, shell=True, check=True)
    
    # Unzip
    print("\n📦 Extracting...")
    subprocess.run(f"cd {cifake_dir} && unzip -q '*.zip' && rm *.zip", shell=True)
    
    print(f"\n✅ CIFAKE downloaded to: {cifake_dir}")

def download_genimage(output_dir):
    """Download GenImage dataset"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING GENIMAGE DATASET")
    print("="*60)
    print("\nSize: ~10GB")
    print("Content: Images from multiple generators")
    print("Generators: StyleGAN, VQGAN, Midjourney, etc")
    print("\n" + "="*60)
    
    if not check_kaggle_setup():
        setup_kaggle()
    
    genimage_dir = Path(output_dir) / "genimage"
    genimage_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Downloading to: {genimage_dir}")
    cmd = f"cd {genimage_dir} && kaggle datasets download -d khaledzsa/genimage"
    subprocess.run(cmd, shell=True, check=True)
    
    print("\n📦 Extracting...")
    subprocess.run(f"cd {genimage_dir} && unzip -q '*.zip' && rm *.zip", shell=True)
    
    print(f"\n✅ GenImage downloaded to: {genimage_dir}")

def download_diffusiondb(output_dir):
    """Download DiffusionDB subset"""
    print("\n" + "="*60)
    print("📥 DOWNLOADING DIFFUSIONDB SUBSET")
    print("="*60)
    print("\nSize: ~5GB (1K images)")
    print("Content: Stable Diffusion generated images")
    print("\n" + "="*60)
    
    if not check_huggingface_setup():
        setup_huggingface()
    
    diffusion_dir = Path(output_dir) / "diffusiondb"
    diffusion_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Downloading to: {diffusion_dir}")
    os.chdir(diffusion_dir)
    subprocess.run([
        "huggingface-cli", "download",
        "--repo-type", "dataset",
        "poloclub/diffusiondb",
        "--include", "2m_first_1k.zip",
        "--local-dir", ".",
        "--resume-download"
    ], check=True)
    
    print("\n📦 Extracting...")
    subprocess.run("unzip -q 2m_first_1k.zip && rm 2m_first_1k.zip", shell=True)
    
    print(f"\n✅ DiffusionDB downloaded to: {diffusion_dir}")

def show_menu():
    """Display dataset selection menu"""
    print("\n" + "="*60)
    print("🎯 PILIH DATASET ALTERNATIF")
    print("="*60)
    
    for key, info in DATASETS.items():
        star = " ⭐" if info['recommended'] else ""
        print(f"\n{key}. {info['name']}{star}")
        print(f"   Size: {info['size']}")
        print(f"   {info['description']}")
    
    print("\n" + "="*60)
    print("⭐ = Recommended for beginners")
    print("="*60)

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚀 ALTERNATIVE DATASET DOWNLOADER")
    print("="*60)
    print("\nDataset alternatif untuk deteksi gambar AI")
    print("Lebih kecil dan praktis dibanding ImagiNet")
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Show menu
    show_menu()
    
    # Get choice
    choice = input("\n❓ Pilih dataset (1-4) atau 'q' untuk quit: ").strip()
    
    if choice.lower() == 'q':
        print("👋 Bye!")
        return
    
    if choice not in DATASETS:
        print("❌ Pilihan tidak valid!")
        return
    
    dataset = DATASETS[choice]
    
    # Download based on choice
    try:
        if choice == "1":
            download_cifake(output_dir)
        elif choice == "2":
            download_genimage(output_dir)
        elif choice == "3":
            download_diffusiondb(output_dir)
        elif choice == "4":
            print("\n" + "="*60)
            print(f"📋 RAISE Dataset (Manual Download)")
            print("="*60)
            print(f"\n1. Kunjungi: {dataset['url']}")
            print("2. Isi form untuk request akses")
            print("3. Download manual setelah dapat link")
            print(f"4. Extract ke: {output_dir}/raise")
            print("\n" + "="*60)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTips:")
        print("- Pastikan koneksi internet stabil")
        print("- Cek space disk cukup")
        print("- Baca error message di atas untuk detail")
        sys.exit(1)

if __name__ == "__main__":
    main()

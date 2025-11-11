import os
import subprocess
import sys

# Target direktori output
OUT_DIR = "data/raw"
os.makedirs(f"{OUT_DIR}/imaginet", exist_ok=True)


def download_imaginet():
    """Download ImagiNet dataset dari HuggingFace"""
    print("\n" + "=" * 60)
    print("🚀 IMAGINET DATASET DOWNLOADER")
    print("=" * 60)
    print("\n⚠️  PERHATIAN: ImagiNet adalah GATED DATASET")
    print("=" * 60)
    print("\nAnda perlu:")
    print("1. Punya akun HuggingFace (https://huggingface.co/join)")
    print("2. Request access: https://huggingface.co/datasets/delyanboychev/imaginet")
    print("3. Login dengan HuggingFace CLI")
    print("\n" + "=" * 60 + "\n")
    
    try:
        # Check if huggingface-cli is installed
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("⚠️  huggingface-cli tidak ditemukan. Menginstall...")
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], check=True)
            print("✅ huggingface-hub berhasil diinstall!\n")
        
        # Check if user is logged in
        whoami_result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True
        )
        
        if whoami_result.returncode != 0:
            print("🔐 Anda belum login ke HuggingFace!")
            print("\n" + "=" * 60)
            print("LANGKAH-LANGKAH LOGIN:")
            print("=" * 60)
            print("\n1. Buat token di: https://huggingface.co/settings/tokens")
            print("2. Copy token yang dibuat")
            print("3. Jalankan: huggingface-cli login")
            print("4. Paste token saat diminta")
            print("5. Jalankan script ini lagi\n")
            
            choice = input("Mau login sekarang? (y/n): ").strip().lower()
            if choice == 'y':
                print("\n🔄 Membuka login prompt...")
                subprocess.run(["huggingface-cli", "login"])
                print("\n✅ Login selesai! Silakan jalankan script ini lagi.")
                return False
            else:
                print("\n❌ Login dibatalkan. Jalankan 'huggingface-cli login' secara manual.")
                return False
        
        print(f"✅ Logged in as: {whoami_result.stdout.strip()}\n")
        print("📥 Downloading ImagiNet Dataset...")
        print("⏳ Ini akan memakan waktu lama tergantung koneksi internet...")
        print("💾 Ukuran dataset: ~190GB+\n")
        
        imaginet_dir = f"{OUT_DIR}/imaginet"
        
        # Ask if user wants to use mirror
        print("� Opsi Download:")
        print("1. HuggingFace official (global)")
        print("2. HF Mirror (lebih cepat untuk Asia/China)")
        mirror_choice = input("\nPilih (1/2) [default: 1]: ").strip() or "1"
        
        # Set environment variable for mirror if chosen
        env = os.environ.copy()
        if mirror_choice == "2":
            print("\n🌏 Menggunakan HF Mirror endpoint...")
            env["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        print("\n🔄 Memulai download...")
        print("⏳ Ini akan memakan waktu lama (~50GB)...")
        print("💡 Tip: Gunakan Ctrl+C untuk pause, lalu jalankan ulang untuk resume\n")
        
        result = subprocess.run(
            [
                "huggingface-cli",
                "download",
                "delyanboychev/imaginet",
                "--repo-type", "dataset",
                "--local-dir", imaginet_dir,
                "--resume-download"
            ],
            check=True,
            text=True,
            capture_output=False,
            env=env
        )
        
        print(f"\n✅ ImagiNet dataset berhasil didownload ke: {imaginet_dir}/")
        
        # Check if we need to extract 7z files
        seven_z_files = [f for f in os.listdir(imaginet_dir) if f.endswith('.7z.001')]
        
        if seven_z_files:
            print("\n" + "=" * 60)
            print("📦 LANGKAH SELANJUTNYA: EXTRACT DATASET")
            print("=" * 60)
            print("\nDitemukan file .7z yang perlu di-extract.")
            print("\n💡 Jalankan command berikut:")
            print(f"   cd {imaginet_dir}")
            print(f"   7z x imaginet.7z.001")
            print("\n⚠️  Jika belum punya 7z, install dulu:")
            print("   sudo apt install p7zip-full")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error saat download ImagiNet: {e}")
        print("\n💡 Kemungkinan penyebab:")
        print("   1. Belum request access ke dataset (PALING UMUM)")
        print("      → Kunjungi: https://huggingface.co/datasets/delyanboychev/imaginet")
        print("      → Klik 'Request Access' atau 'Agree and Access Repository'")
        print("      → Tunggu approval (biasanya instant)")
        print("\n   2. Token HuggingFace tidak punya permission")
        print("      → Login ulang: huggingface-cli login")
        print("      → Pastikan token punya scope 'read'")
        print("\n   3. Koneksi internet terputus")
        print("      → Jalankan ulang script (download akan resume)")
        print("\n   4. Storage penuh")
        print("      → Cek space: df -h")
        return False
    except FileNotFoundError:
        print("❌ Command tidak ditemukan. Pastikan Python dan pip terinstall dengan benar.")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Download dibatalkan oleh user.")
        print("💡 Jalankan ulang script untuk melanjutkan download.")
        return False


def main():
    """Main function untuk mendownload ImagiNet dataset"""
    download_imaginet()
    
    print("\n" + "=" * 60)
    print("🎉 SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    main()

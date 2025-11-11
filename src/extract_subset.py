import subprocess
import os 
from pathlib import Path
from tqdm import tqdm
import time

def run_command(cmd, desc):
    start = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pbar = tqdm(desc=desc, unit="step", leave=True)
    
    for line in process.stdout:
        if line.strip():
            pbar.update(1)
    process.wait()
    pbar.close()
    
    end = time.time()
    duration = end - start
    
    print(f"⏱️{desc} completed in {duration:.2f} seconds.")
    if process.returncode != 0:
        stderr = process.stderr.read()
        raise RuntimeError(f"Command '{' '.join(cmd)}' failed with error:\n{stderr}")

def extract_subset(
    data_dir="../data/raw/imaginet/data",
    output_dir="../data/processed/imaginet/subset",
    categories=(
        "anime",
        "landscape",
        "faces"
    ),
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_name = "_".join(categories)
    filelist = data_dir / f"filelist_{cat_name}.txt"
    selected = data_dir / f"selected_{cat_name}.txt"
    extracted_log = output_dir / f"extracted_{cat_name}.txt"
    
    # Cek semua part
    parts = sorted(data_dir.glob("imaginet.7z.*"))
    if not parts:
        raise FileNotFoundError("❌ Tidak ditemukan file imaginet.7z.001–010 di folder tersebut.")

    print("📦 Ditemukan bagian arsip:")
    for p in parts:
        print("   •", p.name)

    first_part = parts[0]  # misalnya imaginet.7z.001
    
    # 1. Buat daftar isi arsip
    print(f"\n📄 Membuat daftar isi arsip untuk kategori: {cat_name}")
    print("⏳ Listing archive contents (ini mungkin memakan waktu beberapa menit)...")
    
    # Gunakan 7z untuk list semua file, lalu filter
    list_cmd = f"7z l -slt '{first_part}' | grep '^Path = ' | sed 's/^Path = //' > '{filelist}'"
    run_command(
        ["bash", "-c", list_cmd],
        desc="Listing isi arsip"
    )
    
    # 2. Filter kategori
    print(f"🔍 Memilih kategori: {', '.join(categories)}")
    grep_pattern = "|".join(categories)
    
    # Filter file yang mengandung kategori yang diinginkan
    filter_cmd = f"grep -iE '({grep_pattern})' '{filelist}' > '{selected}' || touch '{selected}'"
    run_command(
        ["bash", "-c", filter_cmd],
        desc="Menyaring file sesuai kategori"
    )
    
    # Check if any files were selected
    with open(selected, 'r') as f:
        selected_count = sum(1 for _ in f)
    
    if selected_count == 0:
        print(f"\n⚠️ WARNING: Tidak ada file yang cocok dengan kategori: {', '.join(categories)}")
        print("💡 Tip: Jalankan '7z l imaginet.7z.001' untuk melihat struktur folder di dalam archive")
        raise ValueError(f"Tidak ditemukan file untuk kategori: {', '.join(categories)}")
    
    print(f"✅ Ditemukan {selected_count} file untuk diekstrak")

    # 3. Ekstraksi subset data
    print("📦 Mengekstrak subset data...")
    run_command(
        [
            "7z", "x", str(data_dir / "imaginet.7z.001"),
            f"-o{output_dir}", f"-i@{selected}", "-y"
        ],
        desc="Ekstraksi file"
    )

    # 4. Buat daftar hasil
    print("🧾 Membuat daftar file hasil ekstraksi...")
    run_command(
        ["bash", "-c", f"find '{output_dir}' -type f > '{extracted_log}'"],
        desc="Menyusun daftar file"
    )

    with open(extracted_log, "r") as f:
        count = sum(1 for _ in f)
    print(f"✅ Ekstraksi selesai. Total file: {count}")

    return {
        "categories": categories,
        "output_dir": str(output_dir),
        "total_files": count,
        "log_file": str(extracted_log)
    }
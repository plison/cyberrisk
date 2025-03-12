import requests
import zipfile
import os
from pathlib import Path
from typing import Optional

from config import NVD_FILEPATH

HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
}

def download_zipfile(year: int, nvd_filepath: Path) -> Optional[Path]:
    url: str = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.zip"
    try:
        response: requests.Response = requests.get(url, stream=True, headers=HEADERS)
        response.raise_for_status()
        filepath: Path = nvd_filepath / f"nvdcve-1.1-{year}.json.zip"

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File for {year} downloaded successfully.")
        return filepath
    except requests.ConnectionError as e:
        print(f"Network connection error for {year}: {e}")
    except requests.Timeout as e:
        print(f"Request timed out for {year}: {e}")
    except requests.HTTPError as e:
        print(f"HTTP error occurred for {year}: {e}")
    except (PermissionError, FileNotFoundError) as e:
        print(f"File system error for {year}: {e}")
    except requests.RequestException as e:
        print(f"Unexpected error downloading file for {year}: {e}")
    return None

def extract_zipfile(zip_filepath: Path, nvd_filepath: Path) -> None:
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(nvd_filepath)
        
        try:
            os.remove(zip_filepath)
        except (PermissionError, FileNotFoundError) as e:
            print(f"Error removing zip file {zip_filepath}: {e}")
        
        json_filepath: Path = nvd_filepath / f"nvdcve-1.1-{zip_filepath.stem.split('-')[-1]}"
        print(f"Data extracted successfully: {json_filepath}")
    except zipfile.BadZipFile as e:
        print(f"Invalid or corrupted zip file {zip_filepath}: {e}")
    except (PermissionError, FileNotFoundError, zipfile.LargeZipFile) as e:
        print(f"File system error with {zip_filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error extracting {zip_filepath}: {e}")

def download_and_process_cve_data(nvd_filepath: Path) -> None:
    try:
        nvd_filepath.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"Error creating directory {nvd_filepath}: {e}")
        return

    for year in range(2002, 2026):
        zip_filepath: Optional[Path] = download_zipfile(year, nvd_filepath)
        if zip_filepath:
            extract_zipfile(zip_filepath, nvd_filepath)

if __name__ == "__main__":
    download_and_process_cve_data(NVD_FILEPATH)
#!/usr/bin/env python3
"""
Tesseract OCR Batch Processing Script for Arabic and English PDFs
Processes all PDF files and creates corresponding .txt files
"""

import os
import sys
import subprocess
import requests
import shutil
from pathlib import Path
import time
from datetime import datetime
import re
import tempfile

# Try to import pdf2image, install if not available
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("pdf2image not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdf2image"])
    try:
        from pdf2image import convert_from_path
        PDF2IMAGE_AVAILABLE = True
    except ImportError:
        PDF2IMAGE_AVAILABLE = False

# Configuration
INPUT_DIR = r"C:\Users\omarw\OneDrive\Desktop\CBJ Test\cbj-scraper\mit_jordan_data\pdfs"
OUTPUT_DIR = r"C:\Users\omarw\OneDrive\Desktop\CBJ Test\cbj-scraper\mit_jordan_data\txt_output"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def print_colored(text, color="white"):
    """Print colored text to console"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m", 
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "gray": "\033[90m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def install_poppler():
    """Try to install poppler-utils for PDF conversion"""
    print_colored("Installing poppler-utils for PDF conversion...", "yellow")
    try:
        # Try to install via winget
        subprocess.run(["winget", "install", "--id", "belm0.poppler_windows"], check=True, capture_output=True)
        print_colored("✓ Poppler installed successfully", "green")
        return True
    except:
        try:
            # Try conda
            subprocess.run(["conda", "install", "-c", "conda-forge", "poppler"], check=True, capture_output=True)
            print_colored("✓ Poppler installed via conda", "green")
            return True
        except:
            print_colored("⚠️  Could not install poppler automatically", "yellow")
            print_colored("Please install poppler manually or use alternative method", "yellow")
            return False

def sanitize_filename(filename):
    """Sanitize filename to avoid issues with special characters"""
    # Remove file extension
    name = os.path.splitext(filename)[0]
    
    # Replace problematic characters with underscores
    # Keep only ASCII letters, numbers, underscores, and hyphens
    sanitized = re.sub(r'[^\w\-_.]', '_', name)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Limit length to avoid Windows path length issues
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized + ".txt"

def check_tesseract():
    """Check if Tesseract is installed and find its path"""
    alternative_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        rf"C:\Users\{os.getenv('USERNAME')}\AppData\Local\Tesseract-OCR\tesseract.exe"
    ]
    
    for path in alternative_paths:
        if os.path.exists(path):
            print_colored(f"✓ Found Tesseract at: {path}", "green")
            return path
    
    print_colored("ERROR: Tesseract OCR not found!", "red")
    print_colored("Please install with: winget install --id UB-Mannheim.TesseractOCR", "yellow")
    sys.exit(1)

def download_arabic_language():
    """Download Arabic language data if not present"""
    tesseract_dir = os.path.dirname(TESSERACT_PATH)
    tessdata_dir = os.path.join(tesseract_dir, "tessdata")
    arabic_file = os.path.join(tessdata_dir, "ara.traineddata")
    
    if not os.path.exists(arabic_file):
        print_colored("Arabic language data not found. Downloading...", "yellow")
        try:
            url = "https://github.com/tesseract-ocr/tessdata/raw/main/ara.traineddata"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(arabic_file, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            print_colored("✓ Arabic language data downloaded", "green")
            return True
        except Exception as e:
            print_colored(f"⚠️  Could not download Arabic language data: {e}", "yellow")
            print_colored("Will proceed with English only", "yellow")
            return False
    return True

def get_available_languages():
    """Get list of available Tesseract languages"""
    try:
        result = subprocess.run([TESSERACT_PATH, "--list-langs"], 
                              capture_output=True, text=True, check=True)
        lines = result.stderr.split('\n')
        languages = [line.strip() for line in lines 
                    if line.strip() and not line.startswith("List of")]
        return languages
    except subprocess.CalledProcessError:
        return ["eng"]  # fallback to English only

def convert_pdf_to_images(pdf_path, temp_dir):
    """Convert PDF to images using pdf2image"""
    try:
        if not PDF2IMAGE_AVAILABLE:
            return None, "pdf2image not available"
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300, fmt='png')
        
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
        
        return image_paths, None
    except Exception as e:
        return None, f"PDF conversion failed: {str(e)}"

def process_pdf(input_path, output_path, language_param, original_filename):
    """Process a single PDF file with Tesseract OCR"""
    temp_dir = None
    
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        print_colored(f"    Converting PDF to images...", "gray")
        
        # Convert PDF to images
        image_paths, error = convert_pdf_to_images(input_path, temp_dir)
        
        if error:
            return False, f"ERROR: {error}"
        
        if not image_paths:
            return False, "ERROR: No images extracted from PDF"
        
        print_colored(f"    Extracted {len(image_paths)} page(s)", "gray")
        
        # Process each page and combine results
        all_text = []
        
        for i, image_path in enumerate(image_paths):
            print_colored(f"    Processing page {i+1}/{len(image_paths)}...", "gray")
            
            # Create temporary output for this page
            temp_output_base = os.path.join(temp_dir, f"page_{i+1}_output")
            
            # Run Tesseract OCR on the image
            cmd = [TESSERACT_PATH, image_path, temp_output_base, "-l", language_param]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            temp_txt_file = temp_output_base + ".txt"
            
            if result.returncode == 0 and os.path.exists(temp_txt_file):
                with open(temp_txt_file, 'r', encoding='utf-8') as f:
                    page_text = f.read().strip()
                    if page_text:
                        all_text.append(f"=== Page {i+1} ===\n{page_text}")
            else:
                all_text.append(f"=== Page {i+1} ===\n[OCR failed for this page]")
        
        # Combine all text and save to final output
        combined_text = "\n\n".join(all_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        file_size = os.path.getsize(output_path)
        return True, f"SUCCESS: Created {os.path.basename(output_path)} ({file_size} bytes, {len(image_paths)} pages)"
        
    except subprocess.TimeoutExpired:
        return False, "ERROR: Timeout (>2 minutes per page)"
    except Exception as e:
        return False, f"EXCEPTION: {str(e)}"
    finally:
        # Clean up temporary files and directory
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print_colored(f"  Warning: Could not clean up temp directory: {e}", "yellow")

def main():
    """Main processing function"""
    print_colored("=== Tesseract OCR Batch Processing Script ===", "green")
    print_colored("Processing Arabic and English PDFs to text files", "yellow")
    print()
    
    # Check dependencies
    if not PDF2IMAGE_AVAILABLE:
        print_colored("ERROR: pdf2image library is required but not available", "red")
        print_colored("Please install with: pip install pdf2image", "yellow")
        sys.exit(1)
    
    # Check Tesseract installation
    global TESSERACT_PATH
    TESSERACT_PATH = check_tesseract()
    
    # Setup language support
    print_colored("Checking language support...", "cyan")
    download_arabic_language()
    
    available_langs = get_available_languages()
    print_colored(f"Available languages: {', '.join(available_langs)}", "cyan")
    
    # Set language parameter
    if "ara" in available_langs:
        language_param = "ara+eng"
        print_colored("Using Arabic + English recognition", "green")
    else:
        language_param = "eng"
        print_colored("Using English only recognition", "yellow")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_colored(f"Created output directory: {OUTPUT_DIR}", "green")
    
    # Get all PDF files
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    
    print("-" * 60)
    print_colored(f"Found {len(pdf_files)} PDF files to process", "cyan")
    print_colored(f"Input Directory: {INPUT_DIR}", "yellow")
    print_colored(f"Output Directory: {OUTPUT_DIR}", "yellow")
    print_colored(f"Tesseract Path: {TESSERACT_PATH}", "yellow")
    print("-" * 60)
    
    # Create filename mapping file
    mapping_file = os.path.join(OUTPUT_DIR, "filename_mapping.txt")
    
    # Process files
    processed_count = 0
    success_count = 0
    error_count = 0
    start_time = datetime.now()
    
    with open(mapping_file, 'w', encoding='utf-8') as mapping:
        mapping.write("Original Filename -> Sanitized Filename\n")
        mapping.write("=" * 60 + "\n")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            processed_count += 1
            input_path = str(pdf_file)
            original_filename = pdf_file.name
            sanitized_filename = sanitize_filename(original_filename)
            output_path = os.path.join(OUTPUT_DIR, sanitized_filename)
            
            # Write mapping to file
            mapping.write(f"{original_filename} -> {sanitized_filename}\n")
            
            print_colored(f"[{i}/{len(pdf_files)}] Processing: {original_filename}", "cyan")
            print_colored(f"  Output: {sanitized_filename}", "gray")
            
            success, message = process_pdf(input_path, output_path, language_param, original_filename)
            
            if success:
                success_count += 1
                print_colored(f"  ✓ {message}", "green")
            else:
                error_count += 1
                print_colored(f"  ✗ {message}", "red")
            
            print()  # Add spacing between files
    
    # Final statistics
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("-" * 60)
    print_colored("TESSERACT OCR BATCH PROCESSING COMPLETED", "green")
    print_colored(f"Total files processed: {processed_count}", "cyan")
    print_colored(f"Successful conversions: {success_count}", "green")
    print_colored(f"Failed conversions: {error_count}", "red")
    print_colored(f"Processing time: {duration}", "yellow")
    print_colored(f"Output directory: {OUTPUT_DIR}", "yellow")
    print_colored(f"Filename mapping: {mapping_file}", "yellow")
    
    if error_count > 0:
        print()
        print_colored("Some files failed to process. This may be because:", "yellow")
        print_colored("- PDF files are corrupted or password protected", "yellow")
        print_colored("- Missing dependencies (poppler-utils)", "yellow")
        print_colored("- File permissions issues", "yellow")
    
    # Show some sample output if successful
    if success_count > 0:
        print()
        print_colored("Sample output from first successful file:", "cyan")
        try:
            txt_files = list(Path(OUTPUT_DIR).glob("*.txt"))
            txt_files = [f for f in txt_files if f.name != "filename_mapping.txt"]
            if txt_files:
                with open(txt_files[0], 'r', encoding='utf-8') as f:
                    sample_lines = [f.readline().strip() for _ in range(5)]
                    sample_content = '\n'.join(line for line in sample_lines if line)
                    print_colored(sample_content[:200] + "...", "gray")
        except Exception as e:
            print_colored(f"Could not read sample: {e}", "gray")
    
    print()
    print_colored("Processing complete! Check the txt_output folder for results.", "green")
    print_colored("Check filename_mapping.txt to see original -> sanitized filename mappings.", "green")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\nProcessing interrupted by user.", "yellow")
        sys.exit(1)
    except Exception as e:
        print_colored(f"\nUnexpected error: {e}", "red")
        sys.exit(1) 
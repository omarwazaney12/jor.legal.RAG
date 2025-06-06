#!/usr/bin/env python3
"""
Simple Tesseract OCR Batch Processing Script for PDFs
Uses PyMuPDF (fitz) for PDF processing - no poppler required
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time
from datetime import datetime
import re
import tempfile

# Try to import PyMuPDF, install if not available
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
    try:
        import fitz
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

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

def determine_language_param(available_langs):
    """Determine the best language parameter based on available languages"""
    # Check for Arabic and English support
    has_arabic = "ara" in available_langs
    has_english = "eng" in available_langs
    
    if has_arabic:
        # Use Arabic only for CBJ documents
        return "ara", "Arabic only"
    elif has_english:
        return "eng", "English only (fallback)"
    else:
        return "eng", "English (fallback)"

def convert_pdf_to_images_pymupdf(pdf_path, temp_dir):
    """Convert PDF to images using PyMuPDF"""
    try:
        if not PYMUPDF_AVAILABLE:
            return None, "PyMuPDF not available"
        
        # Open PDF
        doc = fitz.open(pdf_path)
        
        image_paths = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page to image with higher DPI for better OCR results
            # 300 DPI is recommended for OCR (3x zoom factor)
            mat = fitz.Matrix(3.0, 3.0)  # 3x zoom = 216 DPI, good for Arabic OCR
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG (lossless compression, better for OCR)
            image_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
            pix.save(image_path)
            image_paths.append(image_path)
        
        doc.close()
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
        
        # Convert PDF to images using PyMuPDF
        image_paths, error = convert_pdf_to_images_pymupdf(input_path, temp_dir)
        
        if error:
            return False, f"ERROR: {error}"
        
        if not image_paths:
            return False, "ERROR: No images extracted from PDF"
        
        print_colored(f"    Extracted {len(image_paths)} page(s)", "gray")
        
        # Process each page and combine results
        all_text = []
        
        for i, image_path in enumerate(image_paths):
            print_colored(f"    OCR processing page {i+1}/{len(image_paths)}...", "gray")
            
            # Create temporary output for this page
            temp_output_base = os.path.join(temp_dir, f"page_{i+1}_output")
            
            # Enhanced Tesseract command for Arabic OCR
            cmd = [
                TESSERACT_PATH, 
                image_path, 
                temp_output_base, 
                "-l", language_param,
                "--psm", "3",  # Page Segmentation Mode: Fully automatic page segmentation, but no OSD
                "--oem", "3",  # OCR Engine Mode: Default, based on what is available
                "-c", "preserve_interword_spaces=1",  # Preserve spaces between words
            ]
            
            # Add Arabic-specific configuration
            if "ara" in language_param:
                cmd.extend([
                    "-c", "textord_arabic_numeric=1",  # Handle Arabic numerics
                    "-c", "load_system_dawg=0",        # Don't use English dictionary
                    "-c", "load_freq_dawg=0",          # Don't use English frequency dictionary  
                ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            temp_txt_file = temp_output_base + ".txt"
            
            if result.returncode == 0 and os.path.exists(temp_txt_file):
                with open(temp_txt_file, 'r', encoding='utf-8') as f:
                    page_text = f.read().strip()
                    if page_text:
                        all_text.append(f"=== Page {i+1} ===\n{page_text}")
                    else:
                        all_text.append(f"=== Page {i+1} ===\n[No text detected]")
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
    print_colored("=== Simple Tesseract OCR Batch Processing Script ===", "green")
    print_colored("Processing PDFs to text files using PyMuPDF + Tesseract", "yellow")
    print()
    
    # Check dependencies
    if not PYMUPDF_AVAILABLE:
        print_colored("ERROR: PyMuPDF library is required but not available", "red")
        print_colored("Please install with: pip install PyMuPDF", "yellow")
        sys.exit(1)
    
    # Check Tesseract installation
    global TESSERACT_PATH
    TESSERACT_PATH = check_tesseract()
    
    available_langs = get_available_languages()
    print_colored(f"Available languages: {', '.join(available_langs)}", "cyan")
    
    # Set language parameter using intelligent detection
    language_param, language_description = determine_language_param(available_langs)
    print_colored(f"Using {language_description}", "yellow")
    
    if "ara" in language_param:
        print_colored("✓ Arabic OCR support enabled - perfect for CBJ documents!", "green")
    if "eng" in language_param:
        print_colored("✓ English OCR support enabled", "green")
    
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
        print_colored("- PDF files contain no images/text", "yellow")
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
                    sample_content = f.read()[:300]
                    print_colored(sample_content + "...", "gray")
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
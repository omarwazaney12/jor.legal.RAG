#!/usr/bin/env python3
"""
Test Arabic OCR on a few sample files
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import tempfile
import re

# Try to import PyMuPDF
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

def convert_pdf_to_images_pymupdf(pdf_path, temp_dir):
    """Convert PDF to images using PyMuPDF"""
    try:
        if not PYMUPDF_AVAILABLE:
            return None, "PyMuPDF not available"
        
        # Open PDF
        doc = fitz.open(pdf_path)
        
        image_paths = []
        for page_num in range(len(doc)):  # Process all pages, not just first 2
            page = doc[page_num]
            
            # Render page to image with higher DPI for better OCR results
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

def process_pdf_test(input_path, output_path, original_filename):
    """Process a single PDF file with improved Arabic OCR"""
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
                "-l", "ara",  # Arabic only
                "--psm", "3",  # Page Segmentation Mode: Fully automatic page segmentation
                "--oem", "3",  # OCR Engine Mode: Default, based on what is available
                "-c", "preserve_interword_spaces=1",  # Preserve spaces between words
                "-c", "textord_arabic_numeric=1",     # Handle Arabic numerics
                "-c", "load_system_dawg=0",           # Don't use English dictionary
                "-c", "load_freq_dawg=0",             # Don't use English frequency dictionary  
            ]
            
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
    """Process all PDFs with improved Arabic OCR"""
    print_colored("=== Arabic OCR Batch Processing ===", "green")
    print_colored("Processing all PDFs with optimized Arabic OCR settings", "yellow")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    
    print_colored(f"Found {len(pdf_files)} PDF files to process", "cyan")
    print_colored(f"Input Directory: {INPUT_DIR}", "yellow")
    print_colored(f"Output Directory: {OUTPUT_DIR}", "yellow")
    print("-" * 60)
    
    # Create filename mapping file
    mapping_file = os.path.join(OUTPUT_DIR, "filename_mapping.txt")
    
    # Process files
    processed_count = 0
    success_count = 0
    error_count = 0
    
    with open(mapping_file, 'w', encoding='utf-8') as mapping:
        mapping.write("Original Filename -> Output Filename\n")
        mapping.write("=" * 60 + "\n")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            processed_count += 1
            input_path = str(pdf_file)
            original_filename = pdf_file.name
            # Use sanitized filename like the original script
            output_filename = re.sub(r'[^\w\-_.]', '_', original_filename.replace('.pdf', '.txt'))
            output_filename = re.sub(r'_+', '_', output_filename).strip('_')
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Write mapping to file
            mapping.write(f"{original_filename} -> {output_filename}\n")
            
            print_colored(f"[{i}/{len(pdf_files)}] Processing: {original_filename}", "cyan")
            print_colored(f"  Output: {output_filename}", "gray")
            
            success, message = process_pdf_test(input_path, output_path, original_filename)
            
            if success:
                success_count += 1
                print_colored(f"  ✓ {message}", "green")
            else:
                error_count += 1
                print_colored(f"  ✗ {message}", "red")
            
            print()  # Add spacing between files
    
    # Final statistics
    print("-" * 60)
    print_colored("ARABIC OCR BATCH PROCESSING COMPLETED", "green")
    print_colored(f"Total files processed: {processed_count}", "cyan")
    print_colored(f"Successful conversions: {success_count}", "green")
    print_colored(f"Failed conversions: {error_count}", "red")
    print_colored(f"Output directory: {OUTPUT_DIR}", "yellow")
    print_colored(f"Filename mapping: {mapping_file}", "yellow")
    
    if success_count > 0:
        print()
        print_colored("✓ Arabic OCR processing completed successfully!", "green")
        print_colored("Check the txt_output folder for all converted files.", "green")

if __name__ == "__main__":
    main() 
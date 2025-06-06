#!/usr/bin/env python3
"""
OCR Quality Verification Script
Analyzes the output text files to verify Arabic OCR quality
"""

import os
import re
from pathlib import Path
from collections import Counter

OUTPUT_DIR = r"C:\Users\omarw\OneDrive\Desktop\CBJ Test\cbj-scraper\mit_jordan_data\txt_output"

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

def analyze_arabic_content(text):
    """Analyze Arabic text quality"""
    # Count Arabic characters (Unicode range for Arabic)
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', text))
    
    # Count Latin characters 
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # Count numbers
    numbers = len(re.findall(r'[0-9]', text))
    
    # Count total characters (excluding spaces)
    total_chars = len(re.sub(r'\s', '', text))
    
    # Check for common OCR artifacts - simplified pattern
    artifacts = len(re.findall(r'[^\w\s\u0600-\u06FF\u060C\u061B\u061F\u0640().,;:-]', text))
    
    return {
        'arabic_chars': arabic_chars,
        'latin_chars': latin_chars, 
        'numbers': numbers,
        'total_chars': total_chars,
        'artifacts': artifacts,
        'arabic_percentage': (arabic_chars / total_chars * 100) if total_chars > 0 else 0
    }

def verify_ocr_quality():
    """Main verification function"""
    print_colored("=== OCR Quality Verification Report ===", "green")
    print()
    
    # Get all text files (excluding mapping file)
    txt_files = [f for f in Path(OUTPUT_DIR).glob("*.txt") if f.name != "filename_mapping.txt"]
    
    if not txt_files:
        print_colored("❌ No output files found!", "red")
        return
    
    print_colored(f"📁 Found {len(txt_files)} text files to analyze", "cyan")
    print()
    
    # Statistics 
    total_size = 0
    total_lines = 0
    arabic_percentages = []
    file_sizes = []
    quality_issues = []
    
    # Sample analysis
    print_colored("📊 File Statistics:", "yellow")
    print("-" * 60)
    
    for i, txt_file in enumerate(txt_files[:10]):  # Show first 10 files
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            file_size = txt_file.stat().st_size
            line_count = len(content.split('\n'))
            
            analysis = analyze_arabic_content(content)
            
            total_size += file_size
            total_lines += line_count
            file_sizes.append(file_size)
            arabic_percentages.append(analysis['arabic_percentage'])
            
            # Check for quality issues
            if analysis['arabic_percentage'] < 50:
                quality_issues.append(f"{txt_file.name} - Low Arabic content ({analysis['arabic_percentage']:.1f}%)")
            
            if file_size < 1000:  # Less than 1KB might indicate poor OCR
                quality_issues.append(f"{txt_file.name} - Very small file ({file_size} bytes)")
            
            print(f"📄 {txt_file.name[:50]:<50} | {file_size:>8} bytes | {line_count:>4} lines | Arabic: {analysis['arabic_percentage']:>5.1f}%")
            
        except Exception as e:
            print_colored(f"❌ Error processing {txt_file.name}: {str(e)}", "red")
            quality_issues.append(f"{txt_file.name} - Read error: {str(e)}")
    
    if len(txt_files) > 10:
        print(f"... and {len(txt_files) - 10} more files")
    
    print()
    
    # Check if we have data to process
    if not file_sizes:
        print_colored("❌ No files could be processed successfully!", "red")
        print_colored("Checking directory contents...", "yellow")
        
        for txt_file in txt_files[:5]:
            file_size = txt_file.stat().st_size
            print(f"📄 {txt_file.name}: {file_size} bytes")
        
        return
    
    # Overall statistics
    print_colored("📈 Overall Statistics:", "yellow")
    print("-" * 40)
    print(f"Total files processed: {len(txt_files)}")
    print(f"Successfully analyzed: {len(file_sizes)}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print(f"Average file size: {sum(file_sizes) / len(file_sizes):.0f} bytes")
    print(f"Average Arabic content: {sum(arabic_percentages) / len(arabic_percentages):.1f}%")
    print(f"Total lines: {total_lines:,}")
    print()
    
    # Quality assessment
    print_colored("🔍 Quality Assessment:", "yellow")
    print("-" * 40)
    
    avg_arabic = sum(arabic_percentages) / len(arabic_percentages)
    
    if avg_arabic > 80:
        print_colored("✅ Excellent Arabic OCR quality!", "green")
    elif avg_arabic > 60:
        print_colored("✅ Good Arabic OCR quality", "green")
    elif avg_arabic > 40:
        print_colored("⚠️  Moderate Arabic OCR quality", "yellow")
    else:
        print_colored("❌ Poor Arabic OCR quality", "red")
    
    print()
    
    # Issues
    if quality_issues:
        print_colored("⚠️  Potential Issues Found:", "yellow")
        for issue in quality_issues[:5]:  # Show first 5 issues
            print_colored(f"  • {issue}", "yellow")
        if len(quality_issues) > 5:
            print_colored(f"  ... and {len(quality_issues) - 5} more issues", "gray")
    else:
        print_colored("✅ No major quality issues detected!", "green")
    
    print()
    
    # Sample content preview
    print_colored("📖 Sample Content Preview:", "cyan")
    print("-" * 40)
    
    # Find a medium-sized file for preview
    medium_files = [f for f in txt_files if 5000 < f.stat().st_size < 50000]
    if medium_files:
        sample_file = medium_files[0]
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                sample_content = f.read()[:800]  # First 800 characters
            
            print_colored(f"From: {sample_file.name}", "gray")
            print(sample_content)
            print_colored("...", "gray")
        except Exception as e:
            print_colored(f"Could not read sample: {e}", "red")
    
    print()
    print_colored("✅ OCR Quality Verification Completed", "green")

if __name__ == "__main__":
    verify_ocr_quality() 
# ABBYY FineReader Batch OCR Processing Script
# This script processes all PDF files in the specified directory and creates corresponding .txt files

param(
    [string]$InputDirectory = "C:\Users\omarw\OneDrive\Desktop\CBJ Test\cbj-scraper\mit_jordan_data\pdfs",
    [string]$OutputDirectory = "C:\Users\omarw\OneDrive\Desktop\CBJ Test\cbj-scraper\mit_jordan_data\txt_output",
    [string]$FineReaderPath = "C:\Program Files (x86)\ABBYY FineReader PDF 15\FineReaderCommandLine.exe"
)

# Create output directory if it doesn't exist
if (!(Test-Path -Path $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory -Force
    Write-Host "Created output directory: $OutputDirectory" -ForegroundColor Green
}

# Get all PDF files from the input directory
$pdfFiles = Get-ChildItem -Path $InputDirectory -Filter "*.pdf" -File

Write-Host "Found $($pdfFiles.Count) PDF files to process" -ForegroundColor Cyan
Write-Host "Input Directory: $InputDirectory" -ForegroundColor Yellow
Write-Host "Output Directory: $OutputDirectory" -ForegroundColor Yellow
Write-Host "FineReader Path: $FineReaderPath" -ForegroundColor Yellow
Write-Host ("-" * 50) -ForegroundColor Gray

# Check if ABBYY FineReader is installed
if (!(Test-Path -Path $FineReaderPath)) {
    # Try alternative common paths
    $alternativePaths = @(
        "C:\Program Files\ABBYY FineReader PDF 15\FineReaderCommandLine.exe",
        "C:\Program Files (x86)\ABBYY FineReader 15\FineReaderCommandLine.exe",
        "C:\Program Files\ABBYY FineReader 15\FineReaderCommandLine.exe",
        "C:\Program Files (x86)\ABBYY FineReader PDF\FineReaderCommandLine.exe",
        "C:\Program Files\ABBYY FineReader PDF\FineReaderCommandLine.exe"
    )
    
    $foundPath = $null
    foreach ($path in $alternativePaths) {
        if (Test-Path -Path $path) {
            $foundPath = $path
            break
        }
    }
    
    if ($foundPath) {
        $FineReaderPath = $foundPath
        Write-Host "Found ABBYY FineReader at: $FineReaderPath" -ForegroundColor Green
    } else {
        Write-Host "ERROR: ABBYY FineReader Command Line not found!" -ForegroundColor Red
        Write-Host "Please install ABBYY FineReader or update the FineReaderPath parameter" -ForegroundColor Red
        Write-Host "Common installation paths checked:" -ForegroundColor Yellow
        foreach ($path in $alternativePaths) {
            Write-Host "  - $path" -ForegroundColor Gray
        }
        exit 1
    }
}

# Initialize counters
$processedCount = 0
$successCount = 0
$errorCount = 0
$startTime = Get-Date

# Process each PDF file
foreach ($pdfFile in $pdfFiles) {
    $processedCount++
    $inputPath = $pdfFile.FullName
    $outputFileName = [System.IO.Path]::GetFileNameWithoutExtension($pdfFile.Name) + ".txt"
    $outputPath = Join-Path -Path $OutputDirectory -ChildPath $outputFileName
    
    Write-Progress -Activity "Processing PDFs with OCR" -Status "Processing: $($pdfFile.Name)" -PercentComplete (($processedCount / $pdfFiles.Count) * 100)
    
    Write-Host "[$processedCount/$($pdfFiles.Count)] Processing: $($pdfFile.Name)" -ForegroundColor Cyan
    
    try {
        # ABBYY FineReader Command Line syntax
        # Basic command: FineReaderCommandLine.exe /if[format] input_file /of[format] output_file [options]
        $arguments = @(
            "/ifPDF", "`"$inputPath`"",
            "/ofTXT", "`"$outputPath`"",
            "/lang:Arabic,English",  # Support both Arabic and English languages
            "/ido",                  # Ignore document orientation
            "/quit"                  # Quit after processing
        )
        
        $process = Start-Process -FilePath $FineReaderPath -ArgumentList $arguments -Wait -PassThru -NoNewWindow -RedirectStandardOutput "nul" -RedirectStandardError "error.log"
        
        if ($process.ExitCode -eq 0 -and (Test-Path -Path $outputPath)) {
            $successCount++
            Write-Host "  ✓ SUCCESS: Created $outputFileName" -ForegroundColor Green
        } else {
            $errorCount++
            Write-Host "  ✗ ERROR: Failed to process $($pdfFile.Name)" -ForegroundColor Red
            if (Test-Path "error.log") {
                $errorMsg = Get-Content "error.log" -Raw
                Write-Host "    Error details: $errorMsg" -ForegroundColor Red
            }
        }
    }
    catch {
        $errorCount++
        Write-Host "  ✗ EXCEPTION: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # Add a small delay to prevent overwhelming the system
    Start-Sleep -Milliseconds 100
}

# Final statistics
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Progress -Activity "Processing PDFs with OCR" -Completed

Write-Host ("-" * 50) -ForegroundColor Gray
Write-Host "BATCH OCR PROCESSING COMPLETED" -ForegroundColor Green
Write-Host "Total files processed: $processedCount" -ForegroundColor Cyan
Write-Host "Successful conversions: $successCount" -ForegroundColor Green
Write-Host "Failed conversions: $errorCount" -ForegroundColor Red
Write-Host "Processing time: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host "Output directory: $OutputDirectory" -ForegroundColor Yellow

if ($errorCount -gt 0) {
    Write-Host "`nSome files failed to process. Check the error messages above for details." -ForegroundColor Yellow
}

# Clean up temporary files
if (Test-Path "error.log") {
    Remove-Item "error.log" -Force
}

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 
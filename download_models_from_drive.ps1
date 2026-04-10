param(
    [Parameter(Mandatory = $false)]
    [string]$DriveFolder,

    [Parameter(Mandatory = $false)]
    [string]$Destination = ".",

    [Parameter(Mandatory = $false)]
    [switch]$InstallGdown,

    [Parameter(Mandatory = $false)]
    [switch]$UseManifest,

    [Parameter(Mandatory = $false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Get-DriveFolderId {
    param([string]$InputValue)

    if ($InputValue -match "^[a-zA-Z0-9_-]{20,}$") {
        return $InputValue
    }

    if ($InputValue -match "drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)") {
        return $Matches[1]
    }

    throw "Could not parse Drive folder ID. Pass either full folder URL or folder ID."
}

function Get-DriveFileId {
    param([string]$InputValue)

    if ($InputValue -match "^[a-zA-Z0-9_-]{20,}$") {
        return $InputValue
    }

    if ($InputValue -match "drive\.google\.com/file/d/([a-zA-Z0-9_-]+)") {
        return $Matches[1]
    }

    if ($InputValue -match "[?&]id=([a-zA-Z0-9_-]+)") {
        return $Matches[1]
    }

    throw "Could not parse Drive file ID from link: $InputValue"
}

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }

    throw "Python is not installed or not found in PATH."
}

function Test-GdownInstalled {
    param(
        [string[]]$PyCmd
    )

    $exe = $PyCmd[0]
    $args = @()
    if ($PyCmd.Length -gt 1) {
        $args += $PyCmd[1..($PyCmd.Length - 1)]
    }
    $args += @("-m", "gdown", "--version")

    & $exe @args *> $null
    return ($LASTEXITCODE -eq 0)
}

function Download-FromFolder {
    param(
        [string[]]$PyCmd,
        [string]$FolderInput,
        [string]$DestinationPath
    )

    $folderId = Get-DriveFolderId -InputValue $FolderInput
    $driveUrl = "https://drive.google.com/drive/folders/$folderId"

    Write-Host "Downloading Drive folder to: $DestinationPath"
    Write-Host "Drive folder URL: $driveUrl"

    Push-Location $DestinationPath
    try {
        $exe = $PyCmd[0]
        $downloadArgs = @()
        if ($PyCmd.Length -gt 1) {
            $downloadArgs += $PyCmd[1..($PyCmd.Length - 1)]
        }
        $downloadArgs += @("-m", "gdown", "--folder", $driveUrl, "-O", ".")

        & $exe @downloadArgs

        if ($LASTEXITCODE -ne 0) {
            throw "gdown folder download failed. Check folder permissions and link visibility."
        }

        Write-Host "Folder download completed successfully."
    }
    finally {
        Pop-Location
    }
}

function Download-FromManifest {
    param(
        [string[]]$PyCmd,
        [string]$DestinationPath,
        [bool]$Overwrite
    )

    $manifest = @(
        @{
            RelativePath = "Dadhichi-Track2/88%/best_r2plus1d_qevd.pth"
            Url = "https://drive.google.com/file/d/1m9aK8JgjFa6ewDhLpIb5rAepUybs5veU/view?usp=sharing"
        },
        @{
            RelativePath = "Dadhichi-Track2/88%/latest_checkpoint.pth"
            Url = "https://drive.google.com/file/d/1mbgYeDpvPdjFYlHuuT18oEBN79bWrl3f/view?usp=drive_link"
        },
        @{
            RelativePath = "Dadhichi-Track2/88%/lpcvc_final_unified.onnx"
            Url = "https://drive.google.com/file/d/1tEObF3rGGO69y7DvEM3xeieUcuwLs6WH/view?usp=drive_link"
        },
        @{
            RelativePath = "Dadhichi-Track2/88%/lpcvc_final_unified_fixed.onnx"
            Url = "https://drive.google.com/file/d/1AAq-jPIooA3k5mR1-EL8bs6zXaGCzT_V/view?usp=drive_link"
        },
        @{
            RelativePath = "Dadhichi-Track2/88%/qualcomm_r2plus1d.onnx"
            Url = "https://drive.google.com/file/d/1phVY0DqCkBqSfTdZcvfVa8Kc3nE9zi0s/view?usp=drive_link"
        },
        @{
            RelativePath = "Dadhichi-Track2/88%/qualcomm_r2plus1d.onnx.data"
            Url = "https://drive.google.com/file/d/1-7eo0o6zGjD4h4iEol3C_Y3oXZvnHnDg/view?usp=drive_link"
        },
        @{
            RelativePath = "Dadhichi-Track2/89%/best_r2plus1d_qevd.pth"
            Url = "https://drive.google.com/file/d/1txn4uzy8rdl-XtK6P1KTQkry61diOn1b/view?usp=drive_link"
        },
        @{
            RelativePath = "Dadhichi-Track2/89%/latest_checkpoint.pth"
            Url = "https://drive.google.com/file/d/1X0xFXLdmqNXi-FkyJwH2-L9t9ZoGNrdx/view?usp=drive_link"
        },
        @{
            RelativePath = "Dadhichi-Track2/89%/qualcomm_r2plus1d.onnx"
            Url = "https://drive.google.com/file/d/13QB9_7sFMzYg_AoGcqfbHCG9BthXBw0I/view?usp=drive_link"
        },
        @{
            RelativePath = "Dadhichi-Track2/89%/calibration_inputs.npy"
            Url = "https://drive.google.com/file/d/12N6WmkBA2vg1rpoSe7PkaPOK7jy22Ny7/view?usp=drive_link"
        }
    )

    Write-Host "Downloading files from built-in manifest to: $DestinationPath"
    Write-Host "Total files: $($manifest.Count)"

    $exe = $PyCmd[0]
    $baseArgs = @()
    if ($PyCmd.Length -gt 1) {
        $baseArgs += $PyCmd[1..($PyCmd.Length - 1)]
    }

    $downloaded = 0
    $skipped = 0

    foreach ($item in $manifest) {
        $relativePath = $item.RelativePath -replace "/", [System.IO.Path]::DirectorySeparatorChar
        $targetPath = Join-Path $DestinationPath $relativePath
        $targetDir = Split-Path -Path $targetPath -Parent

        if (-not (Test-Path -LiteralPath $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }

        if ((-not $Overwrite) -and (Test-Path -LiteralPath $targetPath)) {
            Write-Host "[skip] Exists: $relativePath"
            $skipped++
            continue
        }

        $fileId = Get-DriveFileId -InputValue $item.Url
        $directUrl = "https://drive.google.com/uc?id=$fileId"

        Write-Host "[get ] $relativePath"

        $args = @()
        $args += $baseArgs
        $args += @("-m", "gdown", $directUrl, "-O", $targetPath)

        & $exe @args

        if ($LASTEXITCODE -ne 0) {
            throw "gdown failed for $relativePath"
        }

        $downloaded++
    }

    Write-Host "Manifest download completed. Downloaded: $downloaded, Skipped: $skipped"
}

$pythonCmd = Get-PythonCommand

if (-not (Test-GdownInstalled -PyCmd $pythonCmd)) {
    if (-not $InstallGdown) {
        throw "gdown not found. Re-run with -InstallGdown, or install manually: pip install gdown"
    }

    Write-Host "Installing gdown..."
    $exe = $pythonCmd[0]
    $installArgs = @()
    if ($pythonCmd.Length -gt 1) {
        $installArgs += $pythonCmd[1..($pythonCmd.Length - 1)]
    }
    $installArgs += @("-m", "pip", "install", "--upgrade", "gdown")
    & $exe @installArgs
}

if (-not (Test-Path -LiteralPath $Destination)) {
    New-Item -ItemType Directory -Path $Destination | Out-Null
}

$destinationPath = (Resolve-Path -LiteralPath $Destination).Path

$useManifestMode = $UseManifest -or [string]::IsNullOrWhiteSpace($DriveFolder)

if ($useManifestMode) {
    Download-FromManifest -PyCmd $pythonCmd -DestinationPath $destinationPath -Overwrite:$Force
}
else {
    Download-FromFolder -PyCmd $pythonCmd -FolderInput $DriveFolder -DestinationPath $destinationPath
}

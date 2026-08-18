# JARVIS_V1 | Blind Image Restoration

**Input:** 128×128 degraded grayscale
**Pipeline:** Noise/Blur Analysis → Restoration → 2× Super-Resolution
**Model:** Restormer
**Output:** 256×256 restored grayscale
**Format:** `.npy`
**Metrics:** PSNR • SSIM • LPIPS

---

# 🚀 Quick Start — Inference

The **primary entry point for this project is `run.py`**.

A reviewer can clone the repository, create the required **Python 3.10 virtual environment**, install the dependencies, download the test images, and run inference directly.

## Test Images

The test images required for inference are available here:

**[Download Test Images — Google Drive](https://drive.google.com/drive/folders/1XyIfycjtUyzsDVCGg4W2lb2nUcukgk1-?usp=sharing)**

Download the test dataset and extract it to a local directory.

The inference input directory must contain the degraded `.npy` images:

```text
<INPUT_DIR>\
├── 000001.npy
├── 000002.npy
├── 000003.npy
└── ...
```

Input specification:

```text
Resolution : 128 × 128
Channels   : 1
Format     : .npy
Grayscale  : Yes
```

---

# Windows Inference

## 1. Clone the Repository

Open **PowerShell**:

```powershell
git clone https://github.com/kadarkarai474/JARVIS.git
cd JARVIS
```

## 2. Python 3.10 Is Required

The inference environment **must use Python 3.10**.

Check:

```powershell
py -3.10 --version
```

Expected:

```text
Python 3.10.x
```

> Do not create the virtual environment using Python 3.9 or another Python version.

## 3. Create the Python 3.10 Virtual Environment

```powershell
py -3.10 -m venv restormer_env
```

Verify:

```powershell
.\restormer_env\Scripts\python.exe --version
```

Expected:

```text
Python 3.10.x
```

## 4. Install Dependencies

```powershell
.\restormer_env\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\restormer_env\Scripts\python.exe -m pip install -r requirements.txt
```

## 5. Run Inference

`run.py` accepts two arguments:

```text
run.py <INPUT_DIR> <OUTPUT_DIR>
```

Set the Matplotlib backend:

```powershell
$env:MPLBACKEND="Agg"
```

Run:

```powershell
.\restormer_env\Scripts\python.exe run.py "<INPUT_DIR>" "<OUTPUT_DIR>"
```

### Example

If the downloaded test images are located at:

```text
D:\JARVIS_dataset\test\NoisyLR
```

run:

```powershell
.\restormer_env\Scripts\python.exe run.py "D:\JARVIS_dataset\test\NoisyLR" "D:\JARVIS_output"
```

The input and output directories can be located anywhere on the local machine.

---

# Complete Windows Inference Command Sequence

A reviewer can follow the complete workflow in this order:

```powershell
git clone https://github.com/kadarkarai474/JARVIS.git

cd JARVIS

py -3.10 --version

py -3.10 -m venv restormer_env

.\restormer_env\Scripts\python.exe --version

.\restormer_env\Scripts\python.exe -m pip install --upgrade pip

.\restormer_env\Scripts\python.exe -m pip install -r requirements.txt

$env:MPLBACKEND="Agg"

.\restormer_env\Scripts\python.exe run.py "<INPUT_DIR>" "<OUTPUT_DIR>"
```

Example:

```powershell
.\restormer_env\Scripts\python.exe run.py "D:\JARVIS_dataset\test\NoisyLR" "D:\JARVIS_output"
```

---

# Output

The generated restored images are written to the specified output directory.

Example:

```text
D:\JARVIS_output\
├── 000001.npy
├── 000002.npy
├── 000003.npy
└── ...
```

Output specification:

```text
Resolution : 256 × 256
Channels   : 1
Format     : .npy
Grayscale  : Yes
```

---

# Important Environment Note

For **Google Colab**, shell commands use `!`.

Example:

```python
!python run.py /content/JARVIS_dataset/test/NoisyLR /content/JARVIS/output
```

For **Windows PowerShell, Linux, VS Code terminal, and GitHub Codespaces**, remove `!`.

Example:

```powershell
python run.py "<INPUT_DIR>" "<OUTPUT_DIR>"
```

The `!` character is only a Colab/Jupyter command prefix.

---

# Inference Timing

`run.py` performs the complete inference workflow and reports the available inference and end-to-end timing information.

Actual performance depends on:

* GPU
* CPU
* PyTorch version
* CUDA environment
* Batch size
* Storage speed

---

# Evaluation

When ground-truth images are available, the restoration results can be evaluated using:

**PSNR • SSIM • LPIPS • L1 • MSE**

The test-image link above provides the inputs required to reproduce the submitted inference results.

---

# Training

Training instructions are intentionally provided **after the complete inference workflow**.

Training is not required when the trained checkpoint used by the inference pipeline is available.

The training script and training configuration are documented in the final section of this README.

# Run Five82Press experiment
param(
    [string]$ENV_NAME = "csc448-gpu"
)

$CONDA_INFO = conda info --json | ConvertFrom-Json
$ENV_PATH = $CONDA_INFO.envs | Where-Object { (Split-Path $_ -Leaf) -eq $ENV_NAME } | Select-Object -First 1
if (-not $ENV_PATH) { throw "Conda environment '$ENV_NAME' not found. Run: conda create -n $ENV_NAME ..." }
$PYTHON = Join-Path $ENV_PATH "python.exe"
$REPO   = "C:\Users\becks\repos\582FinalProject\pitfalls-of-kv-cache-compression"
$MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Fix Windows SSL cert store bug
$env:SSL_CERT_FILE = (& $PYTHON -c "import certifi; print(certifi.where())")
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
$env:CURL_CA_BUNDLE = $env:SSL_CERT_FILE

# Make the repo root importable so Five82Press.py can be found at the top level
$env:PYTHONPATH = "C:\Users\becks\repos\582FinalProject;$env:PYTHONPATH"

# Install local kvpress over the system-installed version
& $PYTHON -m pip install -e "$REPO\kvpress" --quiet

# Run the experiment
Push-Location $REPO
try {
    & $PYTHON -m kv_cache_compression.experiments.compressed_context_ifeval `
        --model_name_or_path $MODEL `
        --model_name_shorthand "llama_3.2_1b_instruct" `
        --model_cache_dir "$env:USERPROFILE\.cache\huggingface\hub" `
        --press_name five82 `
        --num_prompts 80 `
        --compression_ratio_start 0.0 `
        --compression_ratio_end 0.9 `
        --compression_ratio_steps 10 `
        --analyze_kept_tokens False `
        --max_new_tokens 256
} finally {
    Pop-Location
}
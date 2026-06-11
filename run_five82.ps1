# Run Five82Press experiment

$PYTHON = "C:\Users\becks\anaconda3\envs\csc448-gpu\python.exe"
$REPO   = "C:\Users\becks\repos\582FinalProject\pitfalls-of-kv-cache-compression"
$MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Fix Windows SSL cert store bug
$env:SSL_CERT_FILE = (& $PYTHON -c "import certifi; print(certifi.where())")
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
$env:CURL_CA_BUNDLE = $env:SSL_CERT_FILE

# Install local kvpress (includes Five82Press) over the system-installed version
& $PYTHON -m pip install -e "$REPO\kvpress" --quiet

# Run the experiment
Push-Location $REPO
try {
    & $PYTHON -m kv_cache_compression.experiments.compressed_context_ifeval `
        --model_name_or_path $MODEL `
        --model_name_shorthand "llama_3.2_3b_instruct" `
        --model_cache_dir "$env:USERPROFILE\.cache\huggingface\hub" `
        --press_name five82 `
        --num_prompts 50 `
        --compression_ratio_start 0.0 `
        --compression_ratio_end 0.5 `
        --compression_ratio_steps 3 `
        --analyze_kept_tokens False `
        --max_new_tokens 256
} finally {
    Pop-Location
}
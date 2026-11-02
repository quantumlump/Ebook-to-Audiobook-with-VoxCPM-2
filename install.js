module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install wheel \"setuptools<81\""
        ],
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: "uv pip install -r app/requirements.txt",
      },
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: ".",
        },
      },
    },
    // Pre-download NLTK tokenizers and VoxCPM 2 model weights into local hf_cache
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: {
          HF_HOME: "hf_cache",
          HF_HUB_ENABLE_HF_TRANSFER: "1"
        },
        message: "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')\""
      }
    }
  ],
}

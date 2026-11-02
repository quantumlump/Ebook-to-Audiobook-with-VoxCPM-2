module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: {
          PYTHONUNBUFFERED: "1",
          VOXCPM_PORT: "{{port}}",
          OMNIVOICE_PORT: "{{port}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
          HF_HOME: "hf_cache",
          GRADIO_ANALYTICS_ENABLED: "False",
          TOKENIZERS_PARALLELISM: "false"
        },
        message: ["python -u app/app.py"],
        on: [
          {
            event: "/(http:\\/\\/[0-9.:]+)/",
            done: true,
          },
        ],
      },
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}",
      },
    },
  ],
}

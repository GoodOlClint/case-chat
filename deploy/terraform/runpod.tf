# A RunPod pod running vLLM that serves DiffusionGemma (NVFP4 4-bit) on one
# RTX 5090. The GPU half of the split stack — the Vultr VPS points its web app
# at the endpoint this exposes (see outputs.tf).
resource "runpod_pod" "vllm" {
  name       = var.pod_name
  image_name = var.vllm_image
  cloud_type = var.cloud_type

  gpu_type_ids = var.gpu_type_ids
  gpu_count    = 1

  # The RTX 5090 is sm_120 — require a CUDA 12.8+ host so the image's CUDA
  # runtime initializes (older-driver hosts fail on Blackwell).
  allowed_cuda_versions = var.allowed_cuda_versions
  data_center_ids       = var.data_center_ids

  # Local disk for the vLLM image layers + the ~18GB NVFP4 model cache.
  container_disk_in_gb = var.container_disk_in_gb

  # Expose the OpenAI server via RunPod's HTTPS proxy.
  ports = ["8000/http"]

  # HF token for the gated Gemma repo. Sensitive — it lands in the (gitignored)
  # state, so keep state private.
  env = {
    HUGGING_FACE_HUB_TOKEN = var.hf_token
    HF_TOKEN               = var.hf_token
  }

  # vLLM flags = the DiffusionGemma recipe, kept identical to the diffusion
  # flags in deploy/docker-compose.yml. Passed as a list so each JSON arg is one
  # literal element and the quoting survives.
  docker_start_cmd = [
    "--model", var.model_id,
    "--max-model-len", tostring(var.max_model_len),
    "--max-num-seqs", tostring(var.max_num_seqs),
    "--gpu-memory-utilization", tostring(var.gpu_fraction),
    "--generation-config", "vllm",
    "--hf-overrides", "{\"diffusion_sampler\":\"entropy_bound\",\"diffusion_entropy_bound\":0.1}",
    "--diffusion-config", "{\"canvas_length\": 256}",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "gemma4",
    "--reasoning-parser", "gemma4",
    "--host", "0.0.0.0",
    "--port", "8000",
  ]
}

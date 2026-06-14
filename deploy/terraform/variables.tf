# ── Secrets (sensitive; supply via TF_VAR_* env or a gitignored tfvars) ──────
variable "vultr_api_key" {
  description = "Vultr API key (provider auth)."
  type        = string
  sensitive   = true
}

variable "hf_token" {
  description = "Hugging Face token with the Gemma license accepted (gated repo)."
  type        = string
  sensitive   = true
}

variable "web_auth_secret" {
  description = "case-chat magic-link signing secret (openssl rand -hex 32)."
  type        = string
  sensitive   = true
}

variable "tunnel_token" {
  description = "Cloudflare Tunnel connector token for the app ingress."
  type        = string
  sensitive   = true
}

# ── RunPod / vLLM ────────────────────────────────────────────────────────────
variable "model_id" {
  description = "vLLM model repo id. NVFP4 4-bit fits the 5090 (Blackwell FP4)."
  type        = string
  default     = "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
}

variable "vllm_image" {
  description = "vLLM OpenAI server image with DiffusionGemma support."
  type        = string
  default     = "vllm/vllm-openai:gemma"
}

variable "gpu_type_ids" {
  description = "RunPod GPU type ids to allow (first available is used)."
  type        = list(string)
  default     = ["NVIDIA GeForce RTX 5090"]
}

variable "allowed_cuda_versions" {
  description = "Host CUDA versions to allow. RTX 5090 (sm_120) needs 12.8+. Empty = no constraint."
  type        = list(string)
  default     = ["12.8", "12.9", "13.0"]
}

variable "data_center_ids" {
  description = "Restrict RunPod to specific data centers. Empty = any with stock."
  type        = list(string)
  default     = []
}

variable "cloud_type" {
  description = "RunPod COMMUNITY (cheapest 5090s) or SECURE."
  type        = string
  default     = "COMMUNITY"
}

variable "pod_name" {
  type    = string
  default = "diffusiongemma-5090"
}

variable "container_disk_in_gb" {
  description = "RunPod local disk: vLLM image + the ~18GB NVFP4 model cache + scratch."
  type        = number
  default     = 50
}

variable "max_model_len" {
  type    = number
  default = 32768
}

variable "max_num_seqs" {
  description = "Keep <=4 — diffusion state buffers OOM at higher values."
  type        = number
  default     = 4
}

variable "gpu_fraction" {
  description = "vLLM owns the whole pod GPU, so 0.90 (recipe denoising headroom)."
  type        = number
  default     = 0.90
}

# ── Vultr VPS (app host) ─────────────────────────────────────────────────────
variable "vps_label" {
  type    = string
  default = "case-chat"
}

variable "vps_region" {
  description = "Vultr region id (e.g. dfw, ewr)."
  type        = string
  default     = "dfw"
}

variable "vps_plan" {
  description = "Vultr plan. CPU TEI + Qdrant + web want ~16GB RAM."
  type        = string
  default     = "vhf-4c-16gb"
}

variable "vps_os_name" {
  description = "Exact Vultr OS catalog name to look up."
  type        = string
  default     = "Ubuntu 24.04 LTS x64"
}

variable "ssh_public_key_path" {
  description = "Path to the SSH public key injected at boot."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "vps_provisioning" {
  description = "When true, opens SSH (to admin_ssh_cidr) for setup. Set false after."
  type        = bool
  default     = true
}

variable "admin_ssh_cidr" {
  description = "Source IP allowed to SSH during provisioning (e.g. your.ip.addr.0)."
  type        = string
  default     = "0.0.0.0"
}

variable "admin_ssh_cidr_size" {
  description = "CIDR bits for admin_ssh_cidr (32 = a single IP; 0 = anywhere)."
  type        = number
  default     = 0
}

# ── App bootstrap ────────────────────────────────────────────────────────────
variable "repo_url" {
  description = "Git URL the VPS clones to build/run the app (must be reachable from the VPS)."
  type        = string
}

variable "repo_ref" {
  type    = string
  default = "main"
}

variable "public_base_url" {
  description = "Public HTTPS hostname Cloudflare serves (used to mint magic links)."
  type        = string
}

variable "active_jurisdiction" {
  type    = string
  default = "ar"
}

variable "embeddings_model" {
  type    = string
  default = "Qwen/Qwen3-Embedding-4B"
}

variable "corpus_tarball_url" {
  description = "Optional URL to a .tar.gz of the case corpus (the sqlite builds from it)."
  type        = string
  default     = ""
}

variable "qdrant_tarball_url" {
  description = "Optional URL to a .tar.gz of the prebuilt Qdrant index."
  type        = string
  default     = ""
}

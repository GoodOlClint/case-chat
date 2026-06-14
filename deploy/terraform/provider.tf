# case-chat deployment — split stack provisioned as code:
#   * RunPod  : a 5090 pod serving DiffusionGemma (NVFP4) via vLLM
#   * Vultr   : an Ubuntu VPS running the app (web + Qdrant + CPU embeddings +
#               cloudflared), pointed at the RunPod endpoint
# The Vultr resources mirror the conventions in ~/Source/homelab/terraform.

terraform {
  required_version = ">= 1.6"
  required_providers {
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = "~> 1.0"
    }
    vultr = {
      source  = "vultr/vultr"
      version = "~> 2.29"
    }
  }
}

# Both API keys passed as variables (set via TF_VAR_* env or a gitignored
# tfvars), matching the homelab convention.
provider "runpod" {
  api_key = var.runpod_api_key
}

provider "vultr" {
  api_key = var.vultr_api_key
}

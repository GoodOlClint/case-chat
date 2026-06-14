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

# RunPod reads RUNPOD_API_KEY from the environment.
provider "runpod" {}

# Vultr key passed as a variable (sourced from TF_VAR_vultr_api_key or tfvars),
# matching the homelab convention.
provider "vultr" {
  api_key = var.vultr_api_key
}

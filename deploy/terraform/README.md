# case-chat deployment (Terraform)

Provisions the split stack as code:

- **RunPod** — a `RTX 5090` pod running `vllm/vllm-openai:gemma`, serving NVFP4
  4-bit **DiffusionGemma 26B-A4B** (Blackwell — required for NVFP4).
- **Vultr** — an Ubuntu VPS running the app-side stack
  (`deploy/docker-compose.vps.yml`: web + Qdrant + CPU embeddings + cloudflared),
  bootstrapped to point at the RunPod endpoint. Modeled on the Vultr conventions
  in `~/Source/homelab/terraform`.

A single `apply` builds both and wires the VPS to the GPU box.

## Prerequisites
- Terraform >= 1.6
- A RunPod API key and a Vultr API key
- A Hugging Face token with the **Gemma license accepted**
- A Cloudflare Tunnel token + public hostname for ingress
- (Recommended) a `corpus.tar.gz` and prebuilt `qdrant.tar.gz` in object storage,
  so the VPS doesn't re-embed 114k chunks on CPU — see `qdrant_tarball_url` /
  `corpus_tarball_url`.

## Deploy
```bash
cd deploy/terraform

export RUNPOD_API_KEY="…"          # RunPod provider (env only)
cp terraform.tfvars.example terraform.tfvars   # fill in secrets + repo_url + public_base_url
# (or export TF_VAR_vultr_api_key / TF_VAR_hf_token / TF_VAR_web_auth_secret / TF_VAR_tunnel_token)

terraform init
terraform apply
```
Outputs: `vllm_endpoint`, `vps_reserved_ip`, `web_url`.

### Smoke-test the model first (recommended)
Before trusting the full stack, verify DiffusionGemma serves on the 5090:
```bash
VLLM_ENDPOINT=$(terraform output -raw vllm_endpoint) bash ../try_vllm.sh
```

### Lock down after provisioning
The firewall opens SSH only while `vps_provisioning = true`. Once the box is up,
set `vps_provisioning = false` and re-apply to close port 22 (cloudflared is the
only ingress — outbound-only, no web ports exposed).

## Security notes
- **The RunPod proxy endpoint is public and unauthenticated.** Fine for a trial;
  for an ongoing backend, put the pod + VPS on a Tailscale/WireGuard network and
  point the app at the private address instead of the public proxy.
- Secrets (HF token, auth secret, tunnel token) are baked into the Vultr startup
  script and live in **Terraform state** (gitignored here). Keep state private;
  use an encrypted remote backend if it leaves your machine.
- `admin_ssh_cidr` defaults to anywhere — set it to your IP (`admin_ssh_cidr_size = 32`).

## If RunPod `apply` says no instances are available
5090 stock is thin. Try, in order: widen `gpu_type_ids`, clear `data_center_ids`,
switch `cloud_type` to `SECURE`, or relax `allowed_cuda_versions` to `[]`.

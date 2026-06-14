# RunPod serves an /http port at https://<pod-id>-<port>.proxy.runpod.net.
# If this 404s, run `terraform show` and confirm the pod's exported id is the
# proxy subdomain (some provider versions expose it as a separate attribute).
output "vllm_endpoint" {
  description = "OpenAI base URL for the 5090 vLLM (also baked into the VPS .env)."
  value       = "https://${runpod_pod.vllm.id}-8000.proxy.runpod.net/v1"
}

output "vps_reserved_ip" {
  description = "Stable VPS IP (SSH here during provisioning)."
  value       = vultr_reserved_ip.vps.subnet
}

output "vps_instance_id" {
  value = vultr_instance.vps.id
}

output "web_url" {
  description = "Public app URL (served via the Cloudflare tunnel)."
  value       = var.public_base_url
}

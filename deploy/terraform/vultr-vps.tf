# case-chat app VPS — Ubuntu host running the app-side stack (web + Qdrant +
# CPU embeddings + cloudflared) pointed at the RunPod vLLM endpoint. Patterned
# on ~/Source/homelab/terraform/vultr-vps.tf (OS lookup, ssh key from file,
# boot startup script, reserved IP, two-phase SSH firewall).

data "vultr_os" "ubuntu" {
  filter {
    name   = "name"
    values = [var.vps_os_name]
  }
}

resource "vultr_ssh_key" "deploy" {
  name    = "${var.vps_label}-deploy"
  ssh_key = trimspace(file(pathexpand(var.ssh_public_key_path)))
}

# Stable IP across instance rebuilds (handy for SSH / DNS).
resource "vultr_reserved_ip" "vps" {
  label   = "${var.vps_label}-ip"
  region  = var.vps_region
  ip_type = "v4"
}

# Boot script: install Docker, clone the repo, write the app .env (wired to the
# RunPod endpoint), optionally fetch the prebuilt index, and bring the stack up.
resource "vultr_startup_script" "bootstrap" {
  name = "${var.vps_label}-bootstrap"
  type = "boot"
  script = base64encode(templatefile("${path.module}/bootstrap.sh.tftpl", {
    repo_url            = var.repo_url
    repo_ref            = var.repo_ref
    vllm_endpoint       = "https://${runpod_pod.vllm.id}-8000.proxy.runpod.net/v1"
    vllm_model          = var.model_id
    embeddings_model    = var.embeddings_model
    hf_token            = var.hf_token
    tunnel_token        = var.tunnel_token
    web_auth_secret     = var.web_auth_secret
    public_base_url     = var.public_base_url
    active_jurisdiction = var.active_jurisdiction
    corpus_tarball_url  = var.corpus_tarball_url
    qdrant_tarball_url  = var.qdrant_tarball_url
  }))
}

# ──────────────────────────────────────────────
# Firewall — cloudflared is outbound-only, so the only inbound is SSH, and only
# during provisioning (two-phase, as in homelab).
# ──────────────────────────────────────────────
resource "vultr_firewall_group" "vps" {
  description = "${var.vps_label} — strict inbound (web served via cloudflared)"
}

resource "vultr_firewall_rule" "ssh_provisioning" {
  count             = var.vps_provisioning ? 1 : 0
  firewall_group_id = vultr_firewall_group.vps.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = var.admin_ssh_cidr
  subnet_size       = var.admin_ssh_cidr_size
  port              = "22"
  notes             = "SSH — provisioning only (set vps_provisioning=false after)"
}

resource "vultr_firewall_rule" "icmp" {
  firewall_group_id = vultr_firewall_group.vps.id
  protocol          = "icmp"
  ip_type           = "v4"
  subnet            = "0.0.0.0"
  subnet_size       = 0
  notes             = "ICMP ping — external monitoring"
}

resource "vultr_instance" "vps" {
  label             = var.vps_label
  region            = var.vps_region
  plan              = var.vps_plan
  os_id             = data.vultr_os.ubuntu.id
  firewall_group_id = vultr_firewall_group.vps.id
  script_id         = vultr_startup_script.bootstrap.id
  reserved_ip_id    = vultr_reserved_ip.vps.id
  ssh_key_ids       = [vultr_ssh_key.deploy.id]
  enable_ipv6       = true
  backups           = "disabled"
  activation_email  = false

  # The startup script id churns when its (interpolated) contents change; the
  # script is boot-time bootstrap, so don't recreate the instance over it.
  lifecycle {
    ignore_changes = [script_id]
  }
}

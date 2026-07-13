# External global HTTP(S) load balancer fronting all three Cloud Run
# services. URL map mirrors the helm-chart ingress path routing:
#   - LLM data-plane paths → gateway
#   - UI asset paths → ui
#   - Everything else → backend (management API: /key/*, /user/*, …)
#
# By default the LB serves plain HTTP on port 80. Set var.lb_domains to a
# list of DNS names already pointing at lb_ip and the stack provisions a
# Google-managed SSL cert + 443 forwarding rule, and the 80 forwarding rule
# is rewritten to redirect HTTP→HTTPS via a redirect-only URL map.

locals {
  tls_enabled = length(var.lb_domains) > 0
}

resource "google_compute_global_address" "lb" {
  name   = "${local.name}-lb-ip"
  labels = local.labels
}

# Serverless NEGs — one per Cloud Run service.
resource "google_compute_region_network_endpoint_group" "gateway" {
  name                  = "${local.name}-gateway-neg"
  region                = var.region
  network_endpoint_type = "SERVERLESS"

  cloud_run {
    service = google_cloud_run_v2_service.gateway.name
  }
}

resource "google_compute_region_network_endpoint_group" "backend" {
  name                  = "${local.name}-backend-neg"
  region                = var.region
  network_endpoint_type = "SERVERLESS"

  cloud_run {
    service = google_cloud_run_v2_service.backend.name
  }
}

resource "google_compute_region_network_endpoint_group" "ui" {
  name                  = "${local.name}-ui-neg"
  region                = var.region
  network_endpoint_type = "SERVERLESS"

  cloud_run {
    service = google_cloud_run_v2_service.ui.name
  }
}

# Backend services wrap each NEG.
resource "google_compute_backend_service" "gateway" {
  name                  = "${local.name}-gateway-bs"
  protocol              = "HTTP"
  load_balancing_scheme = "EXTERNAL_MANAGED"

  backend {
    group = google_compute_region_network_endpoint_group.gateway.id
  }
}

resource "google_compute_backend_service" "backend" {
  name                  = "${local.name}-backend-bs"
  protocol              = "HTTP"
  load_balancing_scheme = "EXTERNAL_MANAGED"

  backend {
    group = google_compute_region_network_endpoint_group.backend.id
  }
}

resource "google_compute_backend_service" "ui" {
  name                  = "${local.name}-ui-bs"
  protocol              = "HTTP"
  load_balancing_scheme = "EXTERNAL_MANAGED"

  backend {
    group = google_compute_region_network_endpoint_group.ui.id
  }
}

# URL map. Default → backend (management API). Path matchers route the
# gateway and UI prefixes elsewhere.
resource "google_compute_url_map" "this" {
  name            = local.name
  default_service = google_compute_backend_service.backend.id

  host_rule {
    hosts        = ["*"]
    path_matcher = "main"
  }

  path_matcher {
    name            = "main"
    default_service = google_compute_backend_service.backend.id

    # UI paths (catch them before any /v1/* gateway rules so /favicon.ico
    # and / take precedence).
    path_rule {
      paths   = local.ui_path_prefixes
      service = google_compute_backend_service.ui.id
    }

    # Gateway path prefixes. GCP URL maps cap a path_rule at 10 path globs,
    # so chunk into rules of 10.
    dynamic "path_rule" {
      for_each = { for idx, chunk in chunklist(local.gateway_path_prefixes, 10) : idx => chunk }
      content {
        paths   = path_rule.value
        service = google_compute_backend_service.gateway.id
      }
    }
  }
}

# Permanent HTTP→HTTPS redirect URL map. Only attached to the port-80
# target proxy when TLS is enabled; otherwise the regular path-routing
# URL map is attached to the HTTP proxy and everything stays plaintext.
resource "google_compute_url_map" "https_redirect" {
  count = local.tls_enabled ? 1 : 0
  name  = "${local.name}-redirect"

  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "this" {
  name    = "${local.name}-http"
  url_map = local.tls_enabled ? google_compute_url_map.https_redirect[0].id : google_compute_url_map.this.id

  # Default-deny on the HTTP-only path: TLS is the supported posture.
  # Operators must either supply DNS names or explicitly opt in.
  lifecycle {
    precondition {
      condition     = local.tls_enabled || var.allow_plaintext_lb
      error_message = "LB has no HTTPS forwarding rule. Either set `lb_domains` to a list of DNS names you want a Google-managed cert for, or set `allow_plaintext_lb = true` to opt into HTTP-only (trial / dev only)."
    }
  }
}

resource "google_compute_global_forwarding_rule" "http" {
  name                  = "${local.name}-http"
  ip_protocol           = "TCP"
  port_range            = "80"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.lb.address
  target                = google_compute_target_http_proxy.this.id
  labels                = local.labels
}

# ---------- HTTPS (gated on var.lb_domains) ----------
#
# Google-managed certs require each listed domain to resolve to lb_ip
# *before* the cert provisions; on first apply the cert sits in
# PROVISIONING for ~15-60 min until DNS propagates. The LB starts serving
# 443 immediately, but cert handshakes fail until the managed cert
# transitions to ACTIVE.

resource "google_compute_managed_ssl_certificate" "this" {
  count = local.tls_enabled ? 1 : 0

  # A managed cert's `domains` is immutable, so changing var.lb_domains
  # forces replacement, and the cert is referenced by the HTTPS target
  # proxy — a destroy-then-create replacement fails with
  # `resourceInUseByAnotherResource`. Hashing the domains into the name
  # makes the name change with the domain set, so create_before_destroy
  # builds the new cert + repoints the proxy before deleting the old one.
  name = "${local.name}-cert-${substr(sha1(join(",", var.lb_domains)), 0, 8)}"

  managed {
    domains = var.lb_domains
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_target_https_proxy" "this" {
  count            = local.tls_enabled ? 1 : 0
  name             = "${local.name}-https"
  url_map          = google_compute_url_map.this.id
  ssl_certificates = [google_compute_managed_ssl_certificate.this[0].id]
}

resource "google_compute_global_forwarding_rule" "https" {
  count                 = local.tls_enabled ? 1 : 0
  name                  = "${local.name}-https"
  ip_protocol           = "TCP"
  port_range            = "443"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.lb.address
  target                = google_compute_target_https_proxy.this[0].id
  labels                = local.labels
}

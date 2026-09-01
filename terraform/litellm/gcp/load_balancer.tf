# HTTP(S) load balancer fronting all three Cloud Run services.
# EXTERNAL_MANAGED uses the global external path, INTERNAL_MANAGED uses
# the global (cross-region) internal managed path. URL map mirrors the helm-chart
# ingress path routing:
#   - LLM data-plane paths → gateway
#   - UI asset paths → ui
#   - Everything else → backend (management API: /key/*, /user/*, …)
#
# By default the LB serves plain HTTP on port 80. Set var.lb_domains to a
# list of DNS names already pointing at lb_ip and the stack provisions a
# Google-managed SSL cert + 443 forwarding rule, and the 80 forwarding rule
# is rewritten to redirect HTTP→HTTPS via a redirect-only URL map.

locals {
  is_external = var.load_balancing_scheme == "EXTERNAL_MANAGED"
  is_internal = var.load_balancing_scheme == "INTERNAL_MANAGED"

  external_tls_enabled = local.is_external && length(var.lb_domains) > 0
  internal_tls_enabled = local.is_internal && length(var.lb_domains) > 0 && length(var.certificate_manager_certificates) > 0
  tls_enabled          = local.external_tls_enabled || local.internal_tls_enabled
}

resource "google_compute_global_address" "lb" {
  count  = local.is_external ? 1 : 0
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

# Backend services wrap each NEG. The selected load_balancing_scheme controls
# whether these serve EXTERNAL_MANAGED or INTERNAL_MANAGED.
resource "google_compute_backend_service" "gateway" {
  name                  = "${local.name}-gateway-bs"
  protocol              = "HTTP"
  load_balancing_scheme = var.load_balancing_scheme

  backend {
    group = google_compute_region_network_endpoint_group.gateway.id
  }
}

resource "google_compute_backend_service" "backend" {
  name                  = "${local.name}-backend-bs"
  protocol              = "HTTP"
  load_balancing_scheme = var.load_balancing_scheme

  backend {
    group = google_compute_region_network_endpoint_group.backend.id
  }
}

resource "google_compute_backend_service" "ui" {
  name                  = "${local.name}-ui-bs"
  protocol              = "HTTP"
  load_balancing_scheme = var.load_balancing_scheme

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
      condition = !local.is_internal || (
        (length(var.lb_domains) == 0 && length(var.certificate_manager_certificates) == 0) ||
        (length(var.lb_domains) > 0 && length(var.certificate_manager_certificates) > 0)
      )
      error_message = "INTERNAL_MANAGED TLS requires both `lb_domains` and `certificate_manager_certificates` to be set (or both empty for HTTP-only)."
    }

    precondition {
      condition     = local.tls_enabled || var.allow_plaintext_lb
      error_message = "LB has no HTTPS forwarding rule. Either set `lb_domains` to a list of DNS names you want a Google-managed cert for, or set `allow_plaintext_lb = true` to opt into HTTP-only (trial / dev only)."
    }
  }
}

resource "google_compute_global_forwarding_rule" "http" {
  count                 = local.is_external ? 1 : 0
  name                  = "${local.name}-http"
  ip_protocol           = "TCP"
  port_range            = "80"
  load_balancing_scheme = var.load_balancing_scheme
  ip_address            = google_compute_global_address.lb[0].address
  target                = google_compute_target_http_proxy.this.id
  labels                = local.labels
}

resource "google_compute_global_forwarding_rule" "http_internal" {
  count                 = local.is_internal ? 1 : 0
  name                  = "${local.name}-http"
  network               = google_compute_network.this.id
  subnetwork            = google_compute_subnetwork.this.id
  ip_protocol           = "TCP"
  port_range            = "80"
  load_balancing_scheme = var.load_balancing_scheme
  target                = google_compute_target_http_proxy.this.id
  labels                = local.labels

  depends_on = [google_compute_subnetwork.managed_proxy]
}

# ---------- HTTPS (gated on var.lb_domains) ----------
#
# Google-managed certs require each listed domain to resolve to lb_ip
# *before* the cert provisions; on first apply the cert sits in
# PROVISIONING for ~15-60 min until DNS propagates. The LB starts serving
# 443 immediately, but cert handshakes fail until the managed cert
# transitions to ACTIVE.

resource "google_compute_managed_ssl_certificate" "this" {
  count = local.external_tls_enabled ? 1 : 0

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
  count                            = local.tls_enabled ? 1 : 0
  name                             = "${local.name}-https"
  url_map                          = google_compute_url_map.this.id
  ssl_certificates                 = local.is_external ? [google_compute_managed_ssl_certificate.this[0].id] : null
  certificate_manager_certificates = local.is_internal ? var.certificate_manager_certificates : null
}

resource "google_compute_global_forwarding_rule" "https" {
  count                 = local.external_tls_enabled ? 1 : 0
  name                  = "${local.name}-https"
  ip_protocol           = "TCP"
  port_range            = "443"
  load_balancing_scheme = var.load_balancing_scheme
  target                = google_compute_target_https_proxy.this[0].id
  labels                = local.labels

  # Configuration for Global External LB
  ip_address            = local.is_external ? google_compute_global_address.lb[0].address : null

  # Configuration for Global Internal LB
  network               = local.is_internal ? google_compute_network.this.id : null
  subnetwork            = local.is_internal ? google_compute_subnetwork.this.id : null

  depends_on = [google_compute_subnetwork.managed_proxy]
}

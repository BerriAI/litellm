locals {
  private_lb         = var.create_runtime && var.private_ingress != null
  private_subnetwork = !local.private_lb ? null : local.create_network ? google_compute_subnetwork.this[0].id : var.private_ingress.subnetwork_id
}

resource "google_compute_subnetwork" "lb_proxy" {
  count = local.private_lb ? (var.private_ingress.proxy_subnet_cidr == null ? 0 : 1) : 0

  name          = "${local.name}-lb-proxy"
  region        = var.region
  network       = local.network_id
  ip_cidr_range = var.private_ingress.proxy_subnet_cidr
  purpose       = "REGIONAL_MANAGED_PROXY"
  role          = "ACTIVE"
}

resource "google_compute_address" "private_lb" {
  count = local.private_lb ? 1 : 0

  name         = "${local.name}-lb-ip"
  region       = var.region
  address_type = "INTERNAL"
  subnetwork   = local.private_subnetwork
}

resource "google_compute_region_backend_service" "private_lb" {
  for_each = local.private_lb ? {
    gateway = google_compute_region_network_endpoint_group.gateway[0].id
    backend = google_compute_region_network_endpoint_group.backend[0].id
    ui      = google_compute_region_network_endpoint_group.ui[0].id
  } : {}

  name                  = "${local.name}-${each.key}-bs"
  region                = var.region
  protocol              = "HTTP"
  load_balancing_scheme = "INTERNAL_MANAGED"

  backend {
    group = each.value
  }
}

resource "google_compute_region_url_map" "private_lb" {
  count = local.private_lb ? 1 : 0

  name            = local.name
  region          = var.region
  default_service = google_compute_region_backend_service.private_lb["backend"].id

  host_rule {
    hosts        = ["*"]
    path_matcher = "main"
  }

  path_matcher {
    name            = "main"
    default_service = google_compute_region_backend_service.private_lb["backend"].id

    path_rule {
      paths   = local.ui_path_prefixes
      service = google_compute_region_backend_service.private_lb["ui"].id
    }

    dynamic "path_rule" {
      for_each = chunklist(local.gateway_path_prefixes, 10)
      content {
        paths   = path_rule.value
        service = google_compute_region_backend_service.private_lb["gateway"].id
      }
    }
  }
}

resource "google_compute_region_target_http_proxy" "private_lb" {
  count = local.private_lb && !local.tls_enabled ? 1 : 0

  name    = "${local.name}-http"
  region  = var.region
  url_map = google_compute_region_url_map.private_lb[0].id
}

resource "google_compute_region_target_https_proxy" "private_lb" {
  count = local.private_lb && local.tls_enabled ? 1 : 0

  name             = "${local.name}-https"
  region           = var.region
  url_map          = google_compute_region_url_map.private_lb[0].id
  ssl_certificates = [var.private_ingress.ssl_certificate_id]
}

resource "google_compute_forwarding_rule" "private_lb" {
  count = local.private_lb ? 1 : 0

  name                  = "${local.name}-internal"
  region                = var.region
  ip_protocol           = "TCP"
  port_range            = local.tls_enabled ? "443" : "80"
  load_balancing_scheme = "INTERNAL_MANAGED"
  network               = local.network_id
  subnetwork            = local.private_subnetwork
  ip_address            = google_compute_address.private_lb[0].address
  target                = local.tls_enabled ? google_compute_region_target_https_proxy.private_lb[0].id : google_compute_region_target_http_proxy.private_lb[0].id
  labels                = local.labels

  depends_on = [google_compute_subnetwork.lb_proxy]
}

resource "google_compute_network" "this" {
  name                    = local.name
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

resource "google_compute_subnetwork" "this" {
  name                     = "${local.name}-${var.region}"
  region                   = var.region
  network                  = google_compute_network.this.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true
}

# Private Services Access (PSA) range for Cloud SQL + Memorystore. Both
# managed services peer with the VPC over the connection below using
# addresses from this range.
resource "google_compute_global_address" "psa" {
  name          = "${local.name}-psa"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.this.id
}

resource "google_service_networking_connection" "psa" {
  network                 = google_compute_network.this.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.psa.name]
}

# Serverless VPC Access connector — required so Cloud Run can reach
# Cloud SQL / Memorystore private IPs via the PSA range.
#
# min/max instances are required by the API now (you can't just set
# machine_type alone). Defaults: 2 e2-micro instances scale up to 3 — fine
# for low-to-moderate Cloud Run egress; bump max if your services push
# heavy private-network traffic.
resource "google_vpc_access_connector" "this" {
  name          = "${local.name}-conn"
  region        = var.region
  network       = google_compute_network.this.name
  ip_cidr_range = var.vpc_connector_cidr
  min_instances = 2
  max_instances = 3
}

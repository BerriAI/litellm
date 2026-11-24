mock_provider "google" {}
mock_provider "google-beta" {}
mock_provider "random" {}

variables {
  project_id     = "test-project"
  region         = "us-central1"
  tenant         = "tenant"
  env            = "test"
  image_registry = "us-central1-docker.pkg.dev/test-project/litellm"
  image_tag      = "v9.9.9"
}

run "private_http_creates_only_a_regional_internal_frontend" {
  command = plan

  variables {
    private_ingress    = { proxy_subnet_cidr = "10.42.0.0/23" }
    allow_plaintext_lb = true
  }

  assert {
    condition = (
      google_compute_forwarding_rule.private_lb[0].load_balancing_scheme == "INTERNAL_MANAGED" &&
      google_compute_forwarding_rule.private_lb[0].port_range == "80" &&
      google_compute_address.private_lb[0].address_type == "INTERNAL" &&
      google_compute_subnetwork.lb_proxy[0].purpose == "REGIONAL_MANAGED_PROXY" &&
      length(google_compute_global_forwarding_rule.http) == 0 &&
      length(google_compute_global_forwarding_rule.https) == 0
    )
    error_message = "Private HTTP must provide a regional internal frontend and proxy subnet, without public forwarding rules."
  }

  assert {
    condition = length(google_compute_region_backend_service.private_lb) == 3 && alltrue([
      for service in google_compute_region_backend_service.private_lb :
      one(service.backend).balancing_mode == "UTILIZATION" && one(service.backend).capacity_scaler == 1
    ])
    error_message = "The gateway, backend, and UI serverless NEGs must use UTILIZATION with full capacity."
  }

  assert {
    condition = alltrue([
      for service in [google_cloud_run_v2_service.gateway[0], google_cloud_run_v2_service.backend[0], google_cloud_run_v2_service.ui[0]] :
      service.ingress == "INGRESS_TRAFFIC_INTERNAL_ONLY" && service.invoker_iam_disabled
      ]) && (
      length(google_cloud_run_v2_service_iam_member.gateway_allusers) +
      length(google_cloud_run_v2_service_iam_member.backend_allusers) +
      length(google_cloud_run_v2_service_iam_member.ui_allusers) == 0
    )
    error_message = "Private Cloud Run services must accept internal load balancer traffic without any allUsers IAM bindings."
  }
}

run "private_https_uses_a_regional_certificate_and_existing_subnets" {
  command = plan

  override_resource {
    target          = google_compute_region_ssl_policy.private_lb[0]
    override_during = plan
    values          = { id = "projects/test-project/regions/us-central1/sslPolicies/private-tls" }
  }

  variables {
    network_id            = "projects/test-project/global/networks/existing"
    create_psa_connection = false
    private_ingress = {
      subnetwork_id      = "projects/test-project/regions/us-central1/subnetworks/existing"
      ssl_certificate_id = "projects/test-project/regions/us-central1/sslCertificates/private-cert"
    }
    lb_domains = ["proxy.internal.example.com"]
  }

  assert {
    condition = (
      google_compute_forwarding_rule.private_lb[0].subnetwork == var.private_ingress.subnetwork_id &&
      google_compute_forwarding_rule.private_lb[0].port_range == "443" &&
      length(google_compute_subnetwork.lb_proxy) == 0 &&
      length(google_compute_region_target_http_proxy.private_lb) == 0 &&
      contains(google_compute_region_target_https_proxy.private_lb[0].ssl_certificates, var.private_ingress.ssl_certificate_id) &&
      output.lb_url == "https://proxy.internal.example.com"
    )
    error_message = "Private HTTPS must reuse the supplied network and certificate, with no HTTP forwarding or new proxy subnet."
  }

  assert {
    condition = (
      google_compute_region_target_https_proxy.private_lb[0].ssl_policy == google_compute_region_ssl_policy.private_lb[0].id &&
      google_compute_region_ssl_policy.private_lb[0].profile == "MODERN" &&
      google_compute_region_ssl_policy.private_lb[0].min_tls_version == "TLS_1_2"
    )
    error_message = "Private HTTPS must attach the modern TLS policy and require at least TLS 1.2."
  }
}

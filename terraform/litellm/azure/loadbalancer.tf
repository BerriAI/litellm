# Application Gateway: the Azure equivalent of AWS ALB. Public-facing;
# path-based routing that mirrors the AWS stack:
#   - LLM data-plane paths (`/v1/*`, `/chat/*`, ...full list in
#     locals.gateway_path_prefixes) -> gateway backend pool
#   - UI asset paths (`/_next/*`, `/assets/*`, ...) -> ui backend pool
#   - everything else (management API) -> backend backend pool
#
# TLS termination happens at the gateway when `key_vault_certificate_id`
# is supplied. Otherwise plaintext is only allowed when
# `allow_plaintext_app_gateway` is true.
#
# Container Apps ingress is internal-only (no external_enabled). The App
# Gateway reaches them via the Container Apps Environment default domain
# (e.g. `<leaf>.azurecontainerapps.io`) on the public DNS.

# ---------- Subnet for the Application Gateway ----------

resource "azurerm_subnet" "app_gateway" {
  name                 = "app-gateway"
  resource_group_name  = local.resource_group_name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 4, 2)] # /20 portion, isolated from container workloads
  service_endpoints    = ["Microsoft.Storage"]
}

resource "azurerm_network_security_group" "app_gateway" {
  name                = "${local.name}-appgw-nsg"
  location            = var.location
  resource_group_name = local.resource_group_name
  tags                = local.tags
}

# App Gateway requires explicit public-inbound rules on ports 80/443.
resource "azurerm_network_security_rule" "app_gateway_http_in" {
  count                       = var.allow_plaintext_app_gateway ? 1 : 0
  name                        = "http-in"
  priority                    = 100
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_range      = "80"
  source_address_prefix       = "*"
  destination_address_prefix  = "*"
  resource_group_name         = local.resource_group_name
  network_security_group_name = azurerm_network_security_group.app_gateway.name
}

resource "azurerm_network_security_rule" "app_gateway_https_in" {
  name                        = "https-in"
  priority                    = 110
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_range      = "443"
  source_address_prefix       = "*"
  destination_address_prefix  = "*"
  resource_group_name         = local.resource_group_name
  network_security_group_name = azurerm_network_security_group.app_gateway.name
}

resource "azurerm_subnet_network_security_group_association" "app_gateway" {
  subnet_id                 = azurerm_subnet.app_gateway.id
  network_security_group_id = azurerm_network_security_group.app_gateway.id
}

# ---------- Public IP ----------

resource "azurerm_public_ip" "app_gateway" {
  name                = "${local.name}-appgw-pip"
  location            = var.location
  resource_group_name = local.resource_group_name
  sku                 = "Standard"
  allocation_method   = "Static"
  tags                = local.tags
}

# ---------- Gateway FQDN lookup ----------
#
# Container Apps with internal ingress expose a private FQDN (only
# resolvable from inside the VNet). We feed those FQDNs into the App
# Gateway backend pools.
locals {
  gateway_fqdn = azurerm_container_app.gateway.ingress[0].fqdn
  backend_fqdn = azurerm_container_app.backend.ingress[0].fqdn
  ui_fqdn      = azurerm_container_app.ui.ingress[0].fqdn
}

# ---------- URL path map ----------
#
# Application Gateway URL path map contains paths -> backend pools. We
# build one rule that combines gateway and ui prefixes into a single
# path rule (Application Gateway allows multiple paths per rule via the
# `paths` list). The default backend is `backend`.
#
# The redirect_listen_priority uses 100 for the path map and 200 for
# the default backend redirect rule.

resource "azurerm_application_gateway" "this" {
  name                = "${local.name}-appgw"
  location            = var.location
  resource_group_name = local.resource_group_name
  tags                = local.tags

  sku {
    name     = "Standard_v2"
    tier     = "Standard_v2"
    capacity = 2
  }

  # WAF v2 is the standard tier's upgrade; this baseline keeps the SKU
  # minimal. Callers needing WAF can swap the SKU without re-creating
  # the gateway (a single azurerm update).
  enable_http2 = true

  frontend_port {
    name = "http"
    port = 80
  }
  frontend_port {
    name = "https"
    port = 443
  }

  # Gateway-level IP config: the subnet the App Gateway instance listens
  # on (a dedicated subnet with `Microsoft.Network/applicationGateways`
  # delegated). Required even when the listener uses public_ip_address_id.
  gateway_ip_configuration {
    name      = "gateway"
    subnet_id = azurerm_subnet.app_gateway.id
  }

  # Frontend (listener-facing) IP config: the public IP that the world
  # reaches the App Gateway at. Read via the `app_gateway_fqdn` output.
  frontend_ip_configuration {
    name                 = "public"
    public_ip_address_id = azurerm_public_ip.app_gateway.id
  }

  # ---------- Backend pools ----------
  backend_address_pool {
    name  = "gateway"
    fqdns = [local.gateway_fqdn]
  }

  backend_address_pool {
    name  = "backend"
    fqdns = [local.backend_fqdn]
  }

  backend_address_pool {
    name  = "ui"
    fqdns = [local.ui_fqdn]
  }

  # ---------- Backend HTTP settings ----------
  backend_http_settings {
    name                  = "gateway-http"
    cookie_based_affinity = "Disabled"
    port                  = 443
    protocol              = "Https"
    request_timeout       = 60
    probe_name            = "gateway-probe"

    host_name = local.gateway_fqdn
  }

  backend_http_settings {
    name                  = "backend-http"
    cookie_based_affinity = "Disabled"
    port                  = 443
    protocol              = "Https"
    request_timeout       = 60
    probe_name            = "backend-probe"

    host_name = local.backend_fqdn
  }

  backend_http_settings {
    name                  = "ui-http"
    cookie_based_affinity = "Disabled"
    port                  = 443
    protocol              = "Https"
    request_timeout       = 60
    probe_name            = "ui-probe"

    host_name = local.ui_fqdn
  }

  # ---------- Health probes ----------
  probe {
    name                = "gateway-probe"
    host                = local.gateway_fqdn
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3

    match {
      status_code = ["200-399"]
    }

    path     = "/health/liveliness"
    protocol = "Https"
  }

  probe {
    name                = "backend-probe"
    host                = local.backend_fqdn
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3

    match {
      status_code = ["200-399"]
    }

    path     = "/health/liveliness"
    protocol = "Https"
  }

  probe {
    name                = "ui-probe"
    host                = local.ui_fqdn
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3

    match {
      status_code = ["200-399"]
    }

    path     = "/"
    protocol = "Https"
  }

  # ---------- Listeners ----------
  http_listener {
    name                           = "http-listener"
    frontend_ip_configuration_name = "public"
    frontend_port_name             = "http"
    protocol                       = "Http"
    host_name                      = null
  }

  http_listener {
    name                           = "https-listener"
    frontend_ip_configuration_name = "public"
    frontend_port_name             = "https"
    protocol                       = "Https"
    ssl_certificate_name           = local.tls_enabled ? "tls" : null
    host_name                      = null
  }

  # ---------- SSL certificate ----------
  ssl_certificate {
    name                = "tls"
    key_vault_secret_id = var.key_vault_certificate_id

    # Only attach the SSL cert when TLS is enabled.
  }

  # ---------- URL path map ----------
  #
  # Single path rule with a default backend -> management API. The rule
  # itself uses a path prefix matcher at "/" with all explicit paths in
  # `local.gateway_path_prefixes` -> gateway backend and the UI paths in
  # `local.ui_path_prefixes` -> ui backend. We leverage the
  # "default_backend_address_pool" for the backend pool; explicit rules
  # override the default.

  url_path_map {
    name                               = "litellm-url-map"
    default_backend_address_pool_name  = "backend"
    default_backend_http_settings_name = "backend-http"

    path_rule {
      name                       = "gateway-prefixes"
      paths                      = local.gateway_path_prefixes
      backend_address_pool_name  = "gateway"
      backend_http_settings_name = "gateway-http"
    }

    path_rule {
      name                       = "ui-prefixes"
      paths                      = concat(local.ui_path_prefixes, local.ui_exact_paths)
      backend_address_pool_name  = "ui"
      backend_http_settings_name = "ui-http"
    }
  }

  # ---------- Request routing rules ----------
  # HTTPS listener routes through the URL path map.
  request_routing_rule {
    name               = "https-routing"
    rule_type          = "PathBasedRouting"
    http_listener_name = "https-listener"
    url_path_map_name  = "litellm-url-map"
    priority           = 100
  }

  # HTTP listener either routes through the URL path map (when plaintext
  # is allowed) or redirects to HTTPS.
  dynamic "request_routing_rule" {
    for_each = var.allow_plaintext_app_gateway ? [1] : []
    content {
      name               = "http-routing"
      rule_type          = "PathBasedRouting"
      http_listener_name = "http-listener"
      url_path_map_name  = "litellm-url-map"
      priority           = 110
    }
  }

  dynamic "redirect_configuration" {
    for_each = var.allow_plaintext_app_gateway ? [] : [1]
    content {
      name                 = "http-to-https"
      redirect_type        = "Permanent"
      target_listener_name = "https-listener"
      include_path         = true
      include_query_string = true
    }
  }

  # Redirect rule for the http listener (only when plaintext is denied).
  dynamic "request_routing_rule" {
    for_each = var.allow_plaintext_app_gateway ? [] : [1]
    content {
      name                        = "http-redirect"
      rule_type                   = "Basic"
      http_listener_name          = "http-listener"
      redirect_configuration_name = "http-to-https"
      priority                    = 200
    }
  }
}

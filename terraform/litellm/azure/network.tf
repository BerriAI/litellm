# Caller-supplied resource group (when `resource_group_name` is set) or a
# freshly-created one. All Azure resources the module creates are scoped
# to the resource group named `local.resource_group_name`.

resource "azurerm_resource_group" "this" {
  count    = var.resource_group_name == "" ? 1 : 0
  name     = local.resource_group_name
  location = var.location
  tags     = local.tags
}

data "azurerm_client_config" "current" {}

# Caller's subscription + tenant for role assignments and Key Vault access
# policies.
data "azurerm_subscription" "current" {}

# ---------- VNet ----------
#
# Subnet allocation:
#   - containers     : /23 (Container Apps Environment infrastructure subnet;
#                            dynamic IP allocation; must be /23 or larger per
#                            Azure docs)
#   - private_endpoints : /24 (private endpoints to Postgres / Redis / Storage
#                                 / Key Vault)

resource "azurerm_virtual_network" "this" {
  name                = "${local.name}-vnet"
  location            = var.location
  resource_group_name = local.resource_group_name
  address_space       = [var.vnet_cidr]
  tags                = local.tags
}

resource "azurerm_subnet" "containers" {
  name                 = "containers"
  resource_group_name  = local.resource_group_name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 4, 0)] # /20 portion, plenty of room for Container Apps dynamic IPs
  service_endpoints    = ["Microsoft.Storage"]

  delegation {
    name = "container-apps"

    service_delegation {
      name = "Microsoft.App/environments"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action",
        "Microsoft.Network/virtualNetworks/subnets/prepareNetworkPolicies/action",
      ]
    }
  }
}

resource "azurerm_subnet" "private_endpoints" {
  name                 = "private-endpoints"
  resource_group_name  = local.resource_group_name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 4, 1)] # /20 portion, separate from containers

  service_endpoints = ["Microsoft.Storage"]
}

# Network Security Group for the Container Apps subnet. Application
# Gateway sits in its own subnet (created below) and reaches the Container
# Apps via the internal ingress FQDN; the gateway subnet's NSG governs
# inbound traffic from the public internet.
resource "azurerm_network_security_group" "containers" {
  name                = "${local.name}-containers-nsg"
  location            = var.location
  resource_group_name = local.resource_group_name
  tags                = local.tags
}

resource "azurerm_subnet_network_security_group_association" "containers" {
  subnet_id                 = azurerm_subnet.containers.id
  network_security_group_id = azurerm_network_security_group.containers.id
}

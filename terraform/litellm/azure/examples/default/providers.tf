provider "azurerm" {
  features {}
  # Default: subscription_id is read from the environment (`ARM_SUBSCRIPTION_ID`).
  # Explicit override: uncomment and set the subscription_id below.
  # subscription_id = "00000000-0000-0000-0000-000000000000"
}

provider "random" {}

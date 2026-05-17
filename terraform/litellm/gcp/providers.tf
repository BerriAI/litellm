provider "google" {
  project = var.project
  region  = var.region
}

provider "google-beta" {
  project = var.project
  region  = var.region
}

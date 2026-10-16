output "target" {
  description = "URL the load generators send to."
  value       = var.gateway_count > 1 ? "http://${aws_lb.gateway[0].dns_name}:4000" : "http://${aws_instance.gateway[0].private_ip}:4000"
}

output "gateway_public_ips" {
  value = aws_instance.gateway[*].public_ip
}

output "mock_public_ips" {
  value = aws_instance.mock[*].public_ip
}

output "mock_private_ips" {
  value = aws_instance.mock[*].private_ip
}

output "loadgen_public_ips" {
  value = aws_instance.loadgen[*].public_ip
}

output "tools_image" {
  value = local.tools_image
}

output "gateway_image" {
  value = local.gateway_image
}

output "ecr_registry" {
  description = "Pass as REGISTRY to scripts/build-images.sh (append the repo prefix shown in the image outputs)."
  value       = local.registry
}

output "master_key" {
  value     = var.master_key
  sensitive = true
}

data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "aws_key_pair" "rig" {
  key_name   = var.name
  public_key = var.ssh_public_key
}

locals {
  template = "${path.module}/templates/node.sh.tftpl"
  common = {
    region           = var.region
    registry         = local.registry
    tools_image      = local.tools_image
    gateway_image    = local.gateway_image
    chat_model       = var.chat_model
    messages_model   = var.messages_model
    chat_groups      = var.chat_groups
    messages_groups  = var.messages_groups
    deployments      = var.deployments_per_group
    chat_provider    = var.chat_provider
    key_count        = var.key_count
    key_seed         = var.key_seed
    master_key       = var.master_key
    mock_faults      = var.mock_faults
    database_url     = local.database_url
    migrations_image = "ghcr.io/berriai/litellm-migrations:v1.86.0-dev"
  }
  # Mocks get fixed addresses so the gateway's config can be written before they boot.
  mock_ips = [for i in range(var.mock_count) : cidrhost(aws_subnet.main.cidr_block, 10 + i)]
  mocks    = join(",", [for ip in local.mock_ips : "http://${ip}:9000"])
}

resource "aws_instance" "mock" {
  count                  = var.mock_count
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.mock_instance_type
  subnet_id              = aws_subnet.main.id
  private_ip             = local.mock_ips[count.index]
  vpc_security_group_ids = [aws_security_group.rig.id]
  placement_group        = aws_placement_group.rig.id
  iam_instance_profile   = aws_iam_instance_profile.node.name
  key_name               = aws_key_pair.rig.key_name
  user_data              = templatefile(local.template, merge(local.common, { role = "mock", mocks = local.mocks }))
  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }
  tags = { Name = "${var.name}-mock-${count.index}" }
}

resource "aws_instance" "gateway" {
  count                  = var.gateway_count
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.gateway_instance_type
  subnet_id              = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.rig.id]
  placement_group        = aws_placement_group.rig.id
  iam_instance_profile   = aws_iam_instance_profile.node.name
  key_name               = aws_key_pair.rig.key_name
  user_data              = templatefile(local.template, merge(local.common, { role = "gateway", mocks = local.mocks }))
  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }
  tags = { Name = "${var.name}-gateway-${count.index}" }
}

resource "aws_instance" "loadgen" {
  count                  = var.loadgen_count
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.loadgen_instance_type
  subnet_id              = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.rig.id]
  placement_group        = aws_placement_group.rig.id
  iam_instance_profile   = aws_iam_instance_profile.node.name
  key_name               = aws_key_pair.rig.key_name
  user_data              = templatefile(local.template, merge(local.common, { role = "loadgen", mocks = local.mocks }))
  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }
  tags = { Name = "${var.name}-loadgen-${count.index}" }
}

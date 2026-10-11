# ECR (image registry), ECS Fargate (runs the container), the load balancer, CloudFront (HTTPS)
# and CloudWatch logs.

# ---------- ECR ----------
resource "aws_ecr_repository" "api" {
  name                 = "${var.name}-api"
  image_tag_mutability = "IMMUTABLE" # a tag always means the same image
  force_delete         = true        # lets `terraform destroy` remove it even with images inside

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 5 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

# ---------- Logs ----------
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.name}-api"
  retention_in_days = 7
}

# ---------- IAM: the execution role lets ECS pull the image, read the 3 secrets and write logs ----------
resource "aws_iam_role" "task_execution" {
  name = "${var.name}-task-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "read_secrets" {
  name = "read-app-secrets"
  role = aws_iam_role.task_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "secretsmanager:GetSecretValue"
      Resource = [
        aws_secretsmanager_secret.database_url.arn,
        aws_secretsmanager_secret.app_api_key.arn,
        aws_secretsmanager_secret.openai_api_key.arn,
      ]
    }]
  })
}

# ---------- ECS ----------
resource "aws_ecs_cluster" "main" {
  name = "${var.name}-cluster"
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512  # 0.5 vCPU
  memory                   = 1024 # 1 GB
  execution_role_arn       = aws_iam_role.task_execution.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64" # matches images built on Apple Silicon; ~20% cheaper than x86
  }

  container_definitions = jsonencode([{
    name         = "api"
    image        = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "OPENAI_MODEL", value = var.openai_model },
      { name = "LANGSMITH_TRACING", value = "false" },
    ]
    secrets = [
      { name = "DATABASE_URL", valueFrom = aws_secretsmanager_secret.database_url.arn },
      { name = "APP_API_KEY", valueFrom = aws_secretsmanager_secret.app_api_key.arn },
      { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.openai_api_key.arn },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "${var.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  health_check_grace_period_seconds = 60

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true # needed to reach OpenAI/ECR without a NAT gateway
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true # a broken new version is rolled back automatically
  }

  depends_on = [aws_lb_listener.http]
}

# ---------- Load balancer: only requests carrying CloudFront's secret header are forwarded ----------
resource "random_password" "origin_verify" {
  length  = 32
  special = false
}

resource "aws_lb" "api" {
  name               = "${var.name}-alb"
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  security_groups    = [aws_security_group.alb.id]
}

resource "aws_lb_target_group" "api" {
  name                 = "${var.name}-api"
  port                 = 8000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "Forbidden"
      status_code  = "403"
    }
  }
}

resource "aws_lb_listener_rule" "from_cloudfront" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 1

  condition {
    http_header {
      http_header_name = "X-Origin-Verify"
      values           = [random_password.origin_verify.result]
    }
  }
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# ---------- CloudFront: free HTTPS address in front of the load balancer ----------
data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled" # an API: never cache answers
}

data "aws_cloudfront_origin_request_policy" "all_viewer_except_host" {
  name = "Managed-AllViewerExceptHostHeader" # forwards X-API-Key, Content-Type and the body
}

resource "aws_cloudfront_distribution" "api" {
  enabled         = true
  comment         = "${var.name} API (HTTPS)"
  price_class     = "PriceClass_100" # cheapest edge locations (North America and Europe)
  is_ipv6_enabled = true

  origin {
    origin_id   = "alb"
    domain_name = aws_lb.api.dns_name
    custom_header {
      name  = "X-Origin-Verify"
      value = random_password.origin_verify.result
    }
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only" # no domain/certificate on the ALB (see README limitations)
      origin_ssl_protocols   = ["TLSv1.2"]
      origin_read_timeout    = 60 # agent requests can take several seconds
    }
  }

  default_cache_behavior {
    target_origin_id         = "alb"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer_except_host.id
    compress                 = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

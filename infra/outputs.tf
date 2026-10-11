output "api_url" {
  description = "The public HTTPS address of the API"
  value       = "https://${aws_cloudfront_distribution.api.domain_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "set_openai_key_command" {
  description = "Run this yourself; it prompts for the key so it never appears in your shell history"
  value       = "read -s -p 'OpenAI key: ' K && aws secretsmanager put-secret-value --secret-id ${aws_secretsmanager_secret.openai_api_key.name} --secret-string \"$K\" --profile ${var.aws_profile} --region ${var.region} && unset K"
}

output "get_app_api_key_command" {
  value = "aws secretsmanager get-secret-value --secret-id ${aws_secretsmanager_secret.app_api_key.name} --query SecretString --output text --profile ${var.aws_profile} --region ${var.region}"
}

output "run_ingest_command" {
  description = "One-off task inside the VPC that loads the docs into RDS"
  value = join(" ", [
    "aws ecs run-task --cluster ${aws_ecs_cluster.main.name} --task-definition ${aws_ecs_task_definition.api.family}",
    "--launch-type FARGATE --profile ${var.aws_profile} --region ${var.region}",
    "--network-configuration 'awsvpcConfiguration={subnets=[${join(",", aws_subnet.public[*].id)}],securityGroups=[${aws_security_group.app.id}],assignPublicIp=ENABLED}'",
    "--overrides '{\"containerOverrides\":[{\"name\":\"api\",\"command\":[\"python\",\"rag/ingest.py\"]}]}'",
  ])
}

output "log_group" {
  value = aws_cloudwatch_log_group.api.name
}

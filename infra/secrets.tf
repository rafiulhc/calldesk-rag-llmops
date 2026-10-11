# Three secrets, injected into the container as environment variables by ECS.
# recovery_window_in_days = 0 deletes them immediately on destroy (otherwise the names stay reserved).

resource "random_password" "app_api_key" {
  length  = 40
  special = false
}

resource "aws_secretsmanager_secret" "database_url" {
  name                    = "${var.name}/DATABASE_URL"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id = aws_secretsmanager_secret.database_url.id
  secret_string = format("postgresql://%s:%s@%s/%s?sslmode=require",
    aws_db_instance.main.username, random_password.db.result,
  aws_db_instance.main.endpoint, aws_db_instance.main.db_name)
}

resource "aws_secretsmanager_secret" "app_api_key" {
  name                    = "${var.name}/APP_API_KEY"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "app_api_key" {
  secret_id     = aws_secretsmanager_secret.app_api_key.id
  secret_string = random_password.app_api_key.result
}

# Created empty on purpose: YOU put the OpenAI key in it with the AWS CLI after `apply`,
# so the key never appears in Terraform code, Terraform state, or this conversation.
resource "aws_secretsmanager_secret" "openai_api_key" {
  name                    = "${var.name}/OPENAI_API_KEY"
  recovery_window_in_days = 0
}

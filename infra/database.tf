# RDS PostgreSQL 17 with pgvector (the extension is created by the app's init_db()).
# Private, encrypted, single-AZ (cheapest) and set up so `terraform destroy` leaves nothing behind.

resource "random_password" "db" {
  length  = 32
  special = false # keeps the connection URL simple to build
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.name}-db"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "main" {
  identifier     = "${var.name}-db"
  engine         = "postgres"
  engine_version = "17.9"
  instance_class = "db.t4g.micro"

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name  = "llmops"
  username = "llmops_admin"
  password = random_password.db.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false
  multi_az               = false

  backup_retention_period = 1
  skip_final_snapshot     = true # otherwise destroy leaves a snapshot that keeps costing money
  deletion_protection     = false
  apply_immediately       = true
}

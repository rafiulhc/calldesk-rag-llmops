# CloudWatch alarms -> SNS topic -> (optional) email.
# The email subscription only starts working after you click the confirmation link AWS emails you.

variable "alarm_email" {
  description = "Where alarm emails go. Leave empty to create the alarms without email."
  type        = string
  default     = ""
}

resource "aws_sns_topic" "alarms" {
  name = "${var.name}-alarms"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alarm_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

locals {
  alb_dimensions = {
    LoadBalancer = aws_lb.api.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
}

# 1. The API is returning errors (500s from the app, e.g. OpenAI or database failures)
resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "${var.name}-api-5xx-errors"
  alarm_description   = "The API returned 5 or more 5xx errors in 5 minutes"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  dimensions          = local.alb_dimensions
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching" # no traffic = no errors
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

# 2. The API is slow (p95 of response time, measured by the load balancer)
resource "aws_cloudwatch_metric_alarm" "api_latency_p95" {
  alarm_name          = "${var.name}-api-p95-latency"
  alarm_description   = "p95 response time above 8 s for 3 of the last 5 minutes (normal: ~2.5 s)"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  dimensions          = local.alb_dimensions
  extended_statistic  = "p95"
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 3
  threshold           = 8
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

# 3. No healthy container behind the load balancer (the API is down)
resource "aws_cloudwatch_metric_alarm" "no_healthy_hosts" {
  count               = var.desired_count > 0 ? 1 : 0 # only meaningful while the API should be running
  alarm_name          = "${var.name}-api-no-healthy-hosts"
  alarm_description   = "No healthy API container for 2 minutes"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HealthyHostCount"
  dimensions          = local.alb_dimensions
  statistic           = "Minimum"
  period              = 60
  evaluation_periods  = 2
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching" # no data while it should be running = something is wrong
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

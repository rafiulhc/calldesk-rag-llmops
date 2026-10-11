terraform {
  required_version = ">= 1.16.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.66.0" # pinned; released 2026-09-21
    }
    random = {
      source  = "hashicorp/random"
      version = "3.9.1"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "calldesk-llmops"
      ManagedBy = "terraform"
    }
  }
}

variable "region" {
  type    = string
  default = "us-west-1"
}

variable "aws_profile" {
  type    = string
  default = "calldesk"
}

variable "name" {
  description = "Prefix for every resource name"
  type        = string
  default     = "calldesk"
}

variable "image_tag" {
  description = "Image tag in ECR to run (tags are immutable, so use a new tag per deploy)"
  type        = string
  default     = "v1"
}

variable "desired_count" {
  description = "Running API containers. Starts at 0: there is no image and no OpenAI key yet."
  type        = number
  default     = 0
}

variable "openai_model" {
  type    = string
  default = "gpt-4o-mini"
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2) # us-west-1a, us-west-1c
}

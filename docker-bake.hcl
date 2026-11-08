variable "IMAGE" {
  default = "marimo-runpod"
}
variable "SOURCE_REVISION" {
  default = "unknown"
}
group "default" {
  targets = ["cpu", "cu126", "cu128", "cu132"]
}
target "common" {
  context = "."
  platforms = ["linux/amd64"]
  args = { SOURCE_REVISION = SOURCE_REVISION }
}
target "cpu" {
  inherits = ["common"]
  args = { RUNTIME_PROFILE = "cpu" }
  tags = ["${IMAGE}:cpu"]
}
target "cu126" {
  inherits = ["common"]
  args = { RUNTIME_PROFILE = "cu126" }
  tags = ["${IMAGE}:cu126"]
}
target "cu128" {
  inherits = ["common"]
  args = { RUNTIME_PROFILE = "cu128" }
  tags = ["${IMAGE}:cu128"]
}
target "cu132" {
  inherits = ["common"]
  args = { RUNTIME_PROFILE = "cu132" }
  tags = ["${IMAGE}:cu132"]
}

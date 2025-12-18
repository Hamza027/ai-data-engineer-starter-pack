$ErrorActionPreference = "Stop"

Write-Host "GET /health"
Invoke-RestMethod "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 10

Write-Host "`nGET /config"
Invoke-RestMethod "http://127.0.0.1:8000/config" | ConvertTo-Json -Depth 10

Write-Host "`nPOST /run-etl"
$body = @{
  country    = "UK"
  min_amount = "100"
  vat_rate   = "0.20"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/run-etl" `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 10

Write-Host "`nGET /runs?limit=3"
Invoke-RestMethod "http://127.0.0.1:8000/runs?limit=3" | ConvertTo-Json -Depth 10

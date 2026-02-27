$projectDir = "C:\Users\NIXIE\Desktop\projects\nixie-trades-bot"
$python = "$projectDir\venv\Scripts\python.exe"
$worker = "$projectDir\mt5_worker.py"

Write-Host "Nixie Trades MT5 Worker - Auto Restart Loop"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

while ($true) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting mt5_worker.py ..."
    & $python $worker
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Worker stopped. Restarting in 5 seconds..."
    Start-Sleep -Seconds 5
}
```

To run it: **Right-click** `run_worker.ps1` → **Run with PowerShell**. If Windows blocks it, open PowerShell as Administrator and type:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
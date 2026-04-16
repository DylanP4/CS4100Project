param(
  [Parameter(Mandatory = $false)]
  [int]$Games = 10,

  [Parameter(Mandatory = $false)]
  [string]$Checkpoint = "data\\ai_agent_experiment.pkl",

  [Parameter(Mandatory = $false)]
  [string]$FailureLog = "data\\llm_failures_experiment.log",

  [Parameter(Mandatory = $false)]
  [switch]$PyVerbose
)

$ErrorActionPreference = "Stop"

# Ensure commands run from repo root even if called elsewhere.
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:CODENAMES_AI_AGENT_PKL = $Checkpoint
$env:CODENAMES_LLM_FAILURE_LOG = $FailureLog

$v = @()
if ($PyVerbose) { $v = @("-v") }

python -m src.llm_selfplay --games $Games @v --checkpoint $Checkpoint --failure-log $FailureLog

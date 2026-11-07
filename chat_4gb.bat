@echo off
rem Chat with Qwen3-235B-A22B through the qms engine, 4 GB memory profile:
rem   --mem-gb 3.75 lets the engine split the budget itself: as many attention layers pinned
rem   as fit, the rest streamed from disk each token, a small expert cache, KV for 1024 tokens.
rem Model path: set QMS_MODEL to the first shard of the GGUF, or edit the default below.
rem Expect roughly 2.6-3 s per generated token. Empty line or /exit quits; Ctrl+C stops.
chcp 65001 >nul
if not defined QMS_MODEL set "QMS_MODEL=E:\models\qwen3-235b\Q4_0\Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf"
"%~dp0bin\qms.exe" chat "%QMS_MODEL%" --mem-gb 3.75 --direct --ctx 1024 --max-new 400 %*

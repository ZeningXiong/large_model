@echo off
rem Chat with Qwen3-235B-A22B through the qms engine, 8 GB memory profile:
rem   trunk resident (4.8 GB) + 2 GB expert cache + KV cache for 2048 tokens  ~= 7.7 GB.
rem Model path: set QMS_MODEL to the first shard of the GGUF, or edit the default below.
rem Extra options are passed through, e.g.:  chat_8gb.bat --system "You are concise." --temp 0.5
rem Type a message and press Enter; an empty line or /exit quits. Ctrl+C stops a long answer.
chcp 65001 >nul
if not defined QMS_MODEL set "QMS_MODEL=E:\models\qwen3-235b\Q4_0\Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf"
"%~dp0bin\qms.exe" chat "%QMS_MODEL%" --stream-experts --direct --cache-gb 2 --ctx 2048 --max-new 512 %*

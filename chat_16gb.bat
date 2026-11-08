@echo off
rem Chat with Qwen3-235B-A22B through the qms engine, 16 GB memory profile:
rem   --mem-gb 15 pins the whole trunk (4.48 GB resident: 94/94 attention layers, embedding
rem   rows read on demand) and gives the rest to the expert cache (~9.7 GB at ctx 2048).
rem   Experts still stream from disk; measured 1.08-1.64 s/token at 16 GB available, about
rem   1.5x faster than the 8 GB profile (bench/16v8_summary.txt).
rem Model path: set QMS_MODEL to the first shard of the GGUF, or edit the default below.
rem Extra options are passed through, e.g.:  chat_16gb.bat --system "You are concise." --temp 0.5
rem On a 16 GB *physical* machine, Windows leaves only ~11-13 GB available; append a lower
rem   budget to override (the last --mem-gb wins), e.g.:  chat_16gb.bat --mem-gb 11.5
chcp 65001 >nul
if not defined QMS_MODEL set "QMS_MODEL=E:\models\qwen3-235b\Q4_0\Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf"
"%~dp0bin\qms.exe" chat "%QMS_MODEL%" --mem-gb 15 --direct --ctx 2048 --max-new 512 %*

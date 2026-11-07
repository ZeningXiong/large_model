# qms: Qwen3-MoE streaming engine (CPU, Windows/MinGW first)
# Build with MSYS2 UCRT64 gcc:  make        Run tests: make test

ifeq ($(origin CC),default)
CC := gcc
endif
# gcc on Windows needs a writable temp dir; MSYS shells sometimes leave TMP unset.
export TMP  ?= $(CURDIR)/.tmp
export TEMP ?= $(CURDIR)/.tmp
$(shell mkdir -p .tmp)
CFLAGS  ?= -O2 -std=gnu11 -Wall -Wextra -Wshadow -Wvla -march=native -fopenmp
LDFLAGS ?= -static -fopenmp -lm -lpsapi

BIN  := bin
SRC  := src
TEST := tests

HDRS := $(wildcard $(SRC)/*.h)
CORE := $(SRC)/gguf.c $(SRC)/quant.c $(SRC)/kernels.c $(SRC)/model.c $(SRC)/tok.c $(SRC)/chat.c

all: $(BIN)/devbw.exe $(BIN)/test_smoke.exe $(BIN)/qms.exe $(BIN)/test_kernels.exe $(BIN)/test_kernels_scalar.exe

$(BIN):
	mkdir -p $(BIN)

$(BIN)/qms.exe: $(SRC)/main.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -I$(SRC) -o $@ $(SRC)/main.c $(CORE) $(LDFLAGS)

$(BIN)/devbw.exe: $(TEST)/devbw.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

$(BIN)/test_smoke.exe: $(TEST)/test_smoke.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -I$(SRC) -o $@ $< $(LDFLAGS)

KSRC := $(TEST)/test_kernels.c $(SRC)/kernels.c $(SRC)/quant.c $(SRC)/gguf.c
$(BIN)/test_kernels.exe: $(KSRC) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -I$(SRC) -o $@ $(KSRC) $(LDFLAGS)

# same kernels compiled without any SIMD, so the portable fallback is tested too
$(BIN)/test_kernels_scalar.exe: $(KSRC) $(HDRS) | $(BIN)
	$(CC) $(filter-out -march=native,$(CFLAGS)) -march=x86-64 -I$(SRC) -o $@ $(KSRC) $(LDFLAGS)

PY   ?= .venv/Scripts/python.exe -I
TINY := tests/fixtures/tiny
P1   := 9707,11,1879,0,151643,374,264,1273
P2   := 100,200,300,400,500,600,700,800,900,1000,1100,1200

# The tiny random-weight test models are generated, not stored in git. Their tokenizer is
# copied from any real Qwen3 GGUF (TOKSRC); llama-quantize comes from a llama.cpp build.
TOKSRC   ?= D:/models/qwen3-30b/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf
QUANTIZE ?= E:/llama.cpp/llama-quantize.exe
fixtures:
	$(PY) tools/make_tiny.py --src $(TOKSRC) --out $(TINY) --quantize $(QUANTIZE)
	$(QUANTIZE) --tensor-type ffn_down_exps=q4_1 $(TINY)/tiny-f32.gguf $(TINY)/tiny-q4_0-down_q4_1.gguf Q4_0

test: all
	./$(BIN)/test_smoke.exe
	./$(BIN)/test_kernels.exe
	./$(BIN)/test_kernels_scalar.exe
	$(PY) tools/test_deq.py $(TINY)/tiny-q4_0-down_q4_1.gguf $(TINY)/tiny-q4_0.gguf
	$(PY) tools/test_tok.py $(TINY)/tiny-q4_0.gguf --random 100 | grep -E 'TOKENIZER PASSED' || exit 1
	$(PY) tools/test_chat.py $(TINY)/tiny-q4_0.gguf | grep -E 'CHAT PASSED' || exit 1
	for f in tiny-f32 tiny-q4_0 tiny-q4_0-down_q4_1; do \
	  for p in $(P1) $(P2); do \
	    echo "exact $$f $$p"; \
	    $(PY) tools/cmp_ref.py $(TINY)/$$f.gguf --ids $$p --gen 16 | grep -E 'VERDICT: PASS' || exit 1; \
	  done; \
	done
	for f in tiny-q4_0 tiny-q4_0-down_q4_1; do \
	  for p in $(P1) $(P2); do \
	    echo "fast $$f $$p"; \
	    $(PY) tools/cmp_ref.py $(TINY)/$$f.gguf --ids $$p --gen 16 --fast --tol 0.6 | grep -E 'VERDICT: PASS' || exit 1; \
	  done; \
	done
	for f in tiny-f32 tiny-q4_0 tiny-q4_0-down_q4_1; do \
	  for x in "" --exact; do \
	    for s in "--cache-gb 1" "--direct" "--direct --cache-gb 0" "--cache-policy lru --cache-gb 0.001" "--direct --prefetch 2 --cache-gb 0.002" "--direct --prefetch 3 --prefetch-extra 4 --cache-gb 0.001" "--pin-layers 0" "--direct --pin-layers 2 --cache-gb 0"; do \
	      echo "stream $$f $$x $$s"; \
	      $(PY) tools/test_stream.py $(TINY)/$$f.gguf --ids $(P1) --gen 16 $$x "--stream-args=$$s" | grep -E 'PASS$$' || exit 1; \
	    done; \
	  done; \
	done

clean:
	rm -rf $(BIN)

.PHONY: all test clean fixtures

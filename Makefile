# Makefile for Qwen3 MoE C inference
CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
LDFLAGS = -lm

# OpenMP support
OPENMP_FLAGS = -fopenmp
OPENMP_LIBS = -fopenmp

# Architecture specific optimizations
ARCH_FLAGS = -march=native -mtune=native

# Debug flags
DEBUG_FLAGS = -g -DDEBUG

# Default target
TARGET = qwen3_moe
SOURCE = qwen_moe.c

# Default build (optimized with OpenMP)
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) $(OPENMP_FLAGS) -o $@ $< $(LDFLAGS) $(OPENMP_LIBS)

# Debug build
debug: $(SOURCE)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(OPENMP_FLAGS) -o $(TARGET)_debug $< $(LDFLAGS) $(OPENMP_LIBS)

# Build without OpenMP (for compatibility)
no-openmp: $(SOURCE)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -o $(TARGET)_no_omp $< $(LDFLAGS)

# Build for older systems (no advanced arch flags)
portable: $(SOURCE)
	$(CC) -O2 -Wall -std=c99 $(OPENMP_FLAGS) -o $(TARGET)_portable $< $(LDFLAGS) $(OPENMP_LIBS)

# Static build
static: $(SOURCE)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) $(OPENMP_FLAGS) -static -o $(TARGET)_static $< $(LDFLAGS) $(OPENMP_LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_no_omp $(TARGET)_portable $(TARGET)_static

# Run tests (requires model file)
test: $(TARGET)
	@echo "Testing with dummy model (will fail without actual model file)"
	@echo "Usage: make MODEL=path/to/model.bin test-model"

test-model: $(TARGET)
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify MODEL=path/to/model.bin"; \
		exit 1; \
	fi
	./$(TARGET) $(MODEL) 0.8 50

# Convert weights (requires Python script)
convert-weights:
	@echo "Converting Qwen3 weights to binary format..."
	@echo "Usage: python convert_qwen3_weights.py Qwen/Qwen3-Coder-30B-A3B-Instruct qwen3_moe.bin"

# Install dependencies for weight conversion
install-deps:
	pip install torch safetensors huggingface_hub

# Help target
help:
	@echo "Available targets:"
	@echo "  $(TARGET)      - Build optimized version with OpenMP (default)"
	@echo "  debug          - Build debug version"
	@echo "  no-openmp      - Build without OpenMP support"
	@echo "  portable       - Build portable version for older systems"
	@echo "  static         - Build statically linked version"
	@echo "  clean          - Remove build artifacts"
	@echo "  test-model     - Test with MODEL=path/to/model.bin"
	@echo "  convert-weights - Show weight conversion command"
	@echo "  install-deps   - Install Python dependencies for weight conversion"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make                    # Build optimized version"
	@echo "  make test-model MODEL=qwen3_moe.bin"
	@echo "  ./$(TARGET) model.bin 0.8 100   # Run with temperature 0.8, max 100 tokens"

.PHONY: debug no-openmp portable static clean test test-model convert-weights install-deps help
# Artemis Makefile - Build automation
# This Makefile provides convenient targets for building, testing, and managing the project

# Configuration
BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CMAKE ?= cmake
CUDA_ARCH ?= native

# Build options
ENABLE_TESTS ?= ON
ENABLE_EXAMPLES ?= ON
ENABLE_HDF5 ?= ON
ENABLE_VISUALIZATION ?= OFF

# Colors for output
COLOR_RESET = \033[0m
COLOR_BOLD = \033[1m
COLOR_GREEN = \033[32m
COLOR_YELLOW = \033[33m
COLOR_BLUE = \033[34m

# Default target
.DEFAULT_GOAL := help

# Phony targets (not actual files)
.PHONY: help all configure build test clean install examples docs format check coverage

##@ General

help: ## Display this help message
	@echo "$(COLOR_BOLD)Artemis Build System$(COLOR_RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(COLOR_BLUE)<target>$(COLOR_RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(COLOR_BLUE)%-15s$(COLOR_RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(COLOR_BOLD)%s$(COLOR_RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

all: build ## Build everything (default: Release)

configure: ## Configure CMake build system
	@echo "$(COLOR_GREEN)Configuring Artemis...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_TESTS=$(ENABLE_TESTS) \
		-DBUILD_EXAMPLES=$(ENABLE_EXAMPLES) \
		-DENABLE_HDF5=$(ENABLE_HDF5) \
		-DBUILD_VISUALIZATION=$(ENABLE_VISUALIZATION) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	@echo "$(COLOR_GREEN)Configuration complete!$(COLOR_RESET)"

build: configure ## Build the project
	@echo "$(COLOR_GREEN)Building Artemis with $(JOBS) parallel jobs...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && $(CMAKE) --build . -j$(JOBS)
	@echo "$(COLOR_GREEN)Build complete!$(COLOR_RESET)"

rebuild: clean build ## Clean and rebuild everything

debug: ## Build with debug symbols
	@$(MAKE) build BUILD_TYPE=Debug

release: ## Build optimized release version
	@$(MAKE) build BUILD_TYPE=Release

##@ Testing

test: build ## Run all tests
	@echo "$(COLOR_GREEN)Running tests...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && ctest --output-on-failure -j$(JOBS)
	@echo "$(COLOR_GREEN)Tests complete!$(COLOR_RESET)"

test-verbose: build ## Run tests with verbose output
	@echo "$(COLOR_GREEN)Running tests (verbose)...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && ctest --verbose --output-on-failure -j$(JOBS)

##@ Examples

examples: build ## Build example programs
	@echo "$(COLOR_GREEN)Building examples...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && $(CMAKE) --build . --target example_basic_boids example_config_based -j$(JOBS)
	@echo "$(COLOR_GREEN)Examples built successfully!$(COLOR_RESET)"

run-basic-boids: examples ## Run basic boids example
	@echo "$(COLOR_GREEN)Running basic boids example...$(COLOR_RESET)"
	@$(BUILD_DIR)/example_basic_boids

run-config-boids: examples ## Run config-based boids example
	@echo "$(COLOR_GREEN)Running config-based example...$(COLOR_RESET)"
	@cp examples/configs/boids.yaml $(BUILD_DIR)/
	@$(BUILD_DIR)/example_config_based $(BUILD_DIR)/boids.yaml

##@ Installation

install: build ## Install Artemis to system
	@echo "$(COLOR_GREEN)Installing Artemis...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && $(CMAKE) --build . --target install
	@echo "$(COLOR_GREEN)Installation complete!$(COLOR_RESET)"

uninstall: ## Uninstall Artemis from system
	@echo "$(COLOR_YELLOW)Uninstalling Artemis...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && $(CMAKE) --build . --target uninstall 2>/dev/null || echo "No uninstall target available"

##@ Maintenance

clean: ## Remove build directory
	@echo "$(COLOR_YELLOW)Cleaning build directory...$(COLOR_RESET)"
	@rm -rf $(BUILD_DIR)
	@echo "$(COLOR_GREEN)Clean complete!$(COLOR_RESET)"

clean-all: clean ## Remove all generated files
	@echo "$(COLOR_YELLOW)Removing all generated files...$(COLOR_RESET)"
	@find . -name "*.o" -delete
	@find . -name "*.so" -delete
	@find . -name "*.a" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf output/ *.csv *.json *.h5 2>/dev/null || true
	@echo "$(COLOR_GREEN)Deep clean complete!$(COLOR_RESET)"

##@ Code Quality

format: ## Format code with clang-format
	@echo "$(COLOR_GREEN)Formatting code...$(COLOR_RESET)"
	@find include src tests examples -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" | \
		xargs clang-format -i -style=file 2>/dev/null || \
		echo "$(COLOR_YELLOW)clang-format not found, skipping...$(COLOR_RESET)"

check: ## Run static analysis
	@echo "$(COLOR_GREEN)Running static analysis...$(COLOR_RESET)"
	@cd $(BUILD_DIR) && $(CMAKE) --build . --target check 2>/dev/null || \
		echo "$(COLOR_YELLOW)No static analysis configured$(COLOR_RESET)"

lint: ## Run linters
	@echo "$(COLOR_GREEN)Running linters...$(COLOR_RESET)"
	@find include src -name "*.hpp" -o -name "*.cpp" | \
		xargs cppcheck --enable=all --suppress=missingIncludeSystem 2>/dev/null || \
		echo "$(COLOR_YELLOW)cppcheck not found, skipping...$(COLOR_RESET)"

##@ Documentation

docs: ## Generate documentation with Doxygen
	@echo "$(COLOR_GREEN)Generating documentation...$(COLOR_RESET)"
	@doxygen Doxyfile 2>/dev/null || echo "$(COLOR_YELLOW)Doxygen not found or not configured$(COLOR_RESET)"

docs-open: docs ## Generate and open documentation
	@xdg-open docs/html/index.html 2>/dev/null || open docs/html/index.html 2>/dev/null || \
		echo "$(COLOR_YELLOW)Could not open documentation$(COLOR_RESET)"

##@ Docker

docker-build: ## Build Docker image
	@echo "$(COLOR_GREEN)Building Docker image...$(COLOR_RESET)"
	@docker build -t artemis:latest .

docker-test: ## Run tests in Docker container
	@echo "$(COLOR_GREEN)Running tests in Docker...$(COLOR_RESET)"
	@docker run --rm --gpus all artemis:latest make test

##@ Information

info: ## Display build configuration
	@echo "$(COLOR_BOLD)Build Configuration:$(COLOR_RESET)"
	@echo "  Build Directory: $(BUILD_DIR)"
	@echo "  Build Type:      $(BUILD_TYPE)"
	@echo "  Parallel Jobs:   $(JOBS)"
	@echo "  Tests:           $(ENABLE_TESTS)"
	@echo "  Examples:        $(ENABLE_EXAMPLES)"
	@echo "  HDF5:            $(ENABLE_HDF5)"
	@echo "  Visualization:   $(ENABLE_VISUALIZATION)"
	@echo ""
	@echo "$(COLOR_BOLD)System Information:$(COLOR_RESET)"
	@echo "  CMake:           $(shell $(CMAKE) --version | head -n1)"
	@echo "  Compiler:        $(shell $(CXX) --version 2>/dev/null | head -n1 || echo "Not found")"
	@echo "  CUDA:            $(shell nvcc --version 2>/dev/null | grep release | cut -d' ' -f5 || echo "Not found")"

version: ## Display Artemis version
	@echo "Artemis v1.0.0"

deps-check: ## Check for required dependencies
	@echo "$(COLOR_BOLD)Checking dependencies:$(COLOR_RESET)"
	@echo -n "  CMake (>=3.18):     "
	@command -v cmake >/dev/null 2>&1 && echo "$(COLOR_GREEN)✓$(COLOR_RESET)" || echo "$(COLOR_YELLOW)✗$(COLOR_RESET)"
	@echo -n "  CUDA (>=11.0):      "
	@command -v nvcc >/dev/null 2>&1 && echo "$(COLOR_GREEN)✓$(COLOR_RESET)" || echo "$(COLOR_YELLOW)✗$(COLOR_RESET)"
	@echo -n "  C++ Compiler:       "
	@command -v g++ >/dev/null 2>&1 && echo "$(COLOR_GREEN)✓$(COLOR_RESET)" || echo "$(COLOR_YELLOW)✗$(COLOR_RESET)"
	@echo -n "  Git:                "
	@command -v git >/dev/null 2>&1 && echo "$(COLOR_GREEN)✓$(COLOR_RESET)" || echo "$(COLOR_YELLOW)✗$(COLOR_RESET)"

##@ Quick Commands

quick: build test ## Quick build and test

dev: ## Setup development build
	@$(MAKE) build BUILD_TYPE=Debug ENABLE_TESTS=ON

ci: clean all test ## Run full CI pipeline locally
	@echo "$(COLOR_GREEN)CI pipeline complete!$(COLOR_RESET)"

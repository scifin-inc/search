// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"os"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestSetLibraryPath validates that SetLibraryPath works correctly
func TestSetLibraryPath(t *testing.T) {
	// Get library path from environment or use default
	libPath := os.Getenv("LLAMA_GO_LIB_PATH")
	if libPath == "" {
		// Try common locations based on OS
		if runtime.GOOS == "windows" {
			libPath = "llama_go.dll"
		} else if runtime.GOOS == "darwin" {
			libPath = "libllama_go.dylib"
		} else {
			libPath = "libllama_go.so"
		}
	}

	// Test: SetLibraryPath should work (idempotent - can be called multiple times)
	// This validates that SetLibraryPath doesn't panic and can be called safely
	assert.NotPanics(t, func() {
		SetLibraryPath(libPath)
		SetLibraryPath(libPath) // Call again to test idempotency
	}, "SetLibraryPath should not panic and should be idempotent")
}

// TestSetLibraryPathWithInvalidPath validates error handling for invalid paths
func TestSetLibraryPathWithInvalidPath(t *testing.T) {
	// Set an invalid library path
	SetLibraryPath("/nonexistent/path/libllama_go.so")

	// Try to create a vectorizer - should fail with file not found error
	_, err := NewVectorizer("test.gguf", 0)
	assert.Error(t, err, "NewVectorizer should fail with invalid library path")
	assert.Contains(t, err.Error(), "library file not found", "Error should mention file not found")
}

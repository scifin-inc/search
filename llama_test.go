// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
BenchmarkLLM/encode-24         	     465	   2305573 ns/op	    2024 B/op	      11 allocs/op
*/
func BenchmarkLLM(b *testing.B) {
	m := loadModel()
	defer m.Close()

	text := "This is a test sentence we are going to generate embeddings for."
	b.Run("encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := m.EmbedText(text)
			assert.NoError(b, err)
		}

	})
}

func loadModel() *Vectorizer {
	// Set library path from environment variable if available
	libPath := os.Getenv("LLAMA_GO_LIB_PATH")
	if libPath == "" {
		// Try to find library in dist directories
		var possiblePaths []string
		if runtime.GOOS == "windows" {
			possiblePaths = []string{"llama_go.dll", "dist/llama_go.dll"}
		} else if runtime.GOOS == "darwin" {
			possiblePaths = []string{"libllama_go.dylib", "dist/libllama_go.dylib"}
		} else {
			// Linux - check architecture-specific directories first
			var archDir string
			if runtime.GOARCH == "arm64" || runtime.GOARCH == "aarch64" {
				archDir = "dist/linux-arm-avx"
			} else {
				archDir = "dist/linux-x64-avx"
			}
			possiblePaths = []string{
				filepath.Join(archDir, "libllama_go.so"), // Architecture-specific first
				"libllama_go.so",
				"dist/libllama_go.so",
				"dist/linux-x64-avx/libllama_go.so",
				"dist/linux-arm-avx/libllama_go.so",
			}
		}

		// Find first existing library file
		for _, path := range possiblePaths {
			if _, err := os.Stat(path); err == nil {
				libPath = path
				break
			}
		}

		// If still not found, use default name
		if libPath == "" {
			if runtime.GOOS == "windows" {
				libPath = "llama_go.dll"
			} else if runtime.GOOS == "darwin" {
				libPath = "libllama_go.dylib"
			} else {
				libPath = "libllama_go.so"
			}
		}
	}
	SetLibraryPath(libPath)

	mod, _ := filepath.Abs("dist/MiniLM-L6-v2.Q8_0.gguf")
	ctx, err := NewVectorizer(mod, 512)
	if err != nil {
		panic(err)
	}
	return ctx
}

func TestEmbedText(t *testing.T) {
	// Check if library file exists before running test
	libPath := os.Getenv("LLAMA_GO_LIB_PATH")
	if libPath == "" {
		// Try to find library in dist directories
		var found bool
		var possiblePaths []string
		if runtime.GOOS == "windows" {
			possiblePaths = []string{"llama_go.dll", "dist/llama_go.dll"}
		} else if runtime.GOOS == "darwin" {
			possiblePaths = []string{"libllama_go.dylib", "dist/libllama_go.dylib"}
		} else {
			// Linux - check architecture-specific directories first
			var archDir string
			if runtime.GOARCH == "arm64" || runtime.GOARCH == "aarch64" {
				archDir = "dist/linux-arm-avx"
			} else {
				archDir = "dist/linux-x64-avx"
			}
			possiblePaths = []string{
				filepath.Join(archDir, "libllama_go.so"), // Architecture-specific first
				"libllama_go.so",
				"dist/libllama_go.so",
				"dist/linux-x64-avx/libllama_go.so",
				"dist/linux-arm-avx/libllama_go.so",
			}
		}

		for _, path := range possiblePaths {
			if _, err := os.Stat(path); err == nil {
				found = true
				break
			}
		}

		if !found {
			t.Skip("Library file not found. Set LLAMA_GO_LIB_PATH environment variable or place library in dist directory.")
		}
	}

	m := loadModel()
	defer m.Close()

	var sb strings.Builder
	for i := 0; i < 10; i++ {
		sb.WriteString("This is a test sentence we are going to generate embeddings for.\n")
	}

	out, err := m.EmbedText(sb.String())
	assert.NoError(t, err)
	assert.NotZero(t, len(out))
}

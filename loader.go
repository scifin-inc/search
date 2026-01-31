// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"fmt"
	"os"
	"sync"

	"github.com/ebitengine/purego"
)

// libptr is a pointer to the loaded dynamic library.
var (
	libptr            uintptr
	libptrMu          sync.Mutex
	libptrInitialized bool
	customLibPath     string
	customLibPathMu   sync.Mutex
)

var load_library func(log_level int) uintptr
var load_model func(path_model string, n_gpu_layers uint32) uintptr
var load_context func(model uintptr, ctx_size uint32, embeddings bool) uintptr
var free_model func(model uintptr)
var free_context func(ctx uintptr)
var embed_size func(model uintptr) int32
var embed_text func(model uintptr, text string, out_embeddings []float32, out_tokens *uint32) int

// SetLibraryPath sets a custom library path to use during initialization.
// Must be called before the first use of NewVectorizer or any library function.
// This is thread-safe and can be called multiple times, but only the first
// call before initialization will take effect.
func SetLibraryPath(libPath string) {
	customLibPathMu.Lock()
	defer customLibPathMu.Unlock()

	libptrMu.Lock()
	defer libptrMu.Unlock()

	if !libptrInitialized {
		customLibPath = libPath
	}
}

// initializeLibrary initializes the library. It's thread-safe and idempotent.
func initializeLibrary() error {
	libptrMu.Lock()
	defer libptrMu.Unlock()

	if libptrInitialized {
		return nil // Already initialized
	}

	// Check if library path was set
	customLibPathMu.Lock()
	libpath := customLibPath
	customLibPathMu.Unlock()

	if libpath == "" {
		return fmt.Errorf("library path not set: call search.SetLibraryPath() before using the library")
	}

	// Verify the file exists
	if _, err := os.Stat(libpath); err != nil {
		return fmt.Errorf("library file not found: %s", libpath)
	}

	// Load the library
	var err error
	libptr, err = load(libpath)
	if err != nil {
		return fmt.Errorf("failed to load library %s: %w", libpath, err)
	}

	// Register functions
	purego.RegisterLibFunc(&load_library, libptr, "load_library")
	purego.RegisterLibFunc(&load_model, libptr, "load_model")
	purego.RegisterLibFunc(&load_context, libptr, "load_context")
	purego.RegisterLibFunc(&free_model, libptr, "free_model")
	purego.RegisterLibFunc(&free_context, libptr, "free_context")
	purego.RegisterLibFunc(&embed_size, libptr, "embed_size")
	purego.RegisterLibFunc(&embed_text, libptr, "embed_text")

	// Initialize the library (Log level WARN)
	load_library(2)

	libptrInitialized = true
	return nil
}

// ensureLibraryLoaded ensures the library is loaded before use
func ensureLibraryLoaded() error {
	if !libptrInitialized {
		return initializeLibrary()
	}
	return nil
}

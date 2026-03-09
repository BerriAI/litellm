// Streaming responses from gollem through LiteLLM.
//
// Uses Go 1.23+ range-over-function iterators for real-time token
// streaming via LiteLLM's SSE passthrough.
//
// Usage:
//
//	litellm --model gpt-4o
//	go run ./streaming
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/fugue-labs/gollem/core"
	"github.com/fugue-labs/gollem/provider/openai"
)

func main() {
	proxyURL := "http://localhost:4000"
	if u := os.Getenv("LITELLM_PROXY_URL"); u != "" {
		proxyURL = u
	}

	model := openai.NewLiteLLM(proxyURL,
		openai.WithModel("gpt-4o"),
	)

	agent := core.NewAgent[string](model)

	// RunStream returns a streaming result that yields tokens as they arrive.
	stream, err := agent.RunStream(context.Background(), "Write a haiku about distributed systems")
	if err != nil {
		log.Fatal(err)
	}

	// StreamText yields text chunks in real-time.
	// The boolean argument controls whether deltas (true) or accumulated
	// text (false) is returned.
	fmt.Print("Response: ")
	for text, err := range stream.StreamText(true) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(text)
	}
	fmt.Println()

	// After streaming completes, the final response is available.
	resp := stream.Response()
	fmt.Printf("\nTokens used: input=%d, output=%d\n",
		resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

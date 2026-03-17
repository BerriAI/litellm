// Basic gollem agent connected to a LiteLLM proxy.
//
// Usage:
//
//	litellm --model gpt-4o   # start proxy in another terminal
//	go run ./basic
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

	// Connect to LiteLLM proxy. NewLiteLLM creates an OpenAI-compatible
	// provider pointed at the given URL.
	model := openai.NewLiteLLM(proxyURL,
		openai.WithModel("gpt-4o"), // any model name configured in LiteLLM
	)

	// Create and run a simple agent.
	agent := core.NewAgent[string](model,
		core.WithSystemPrompt[string]("You are a helpful assistant. Be concise."),
	)

	result, err := agent.Run(context.Background(), "Explain quantum computing in two sentences.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}

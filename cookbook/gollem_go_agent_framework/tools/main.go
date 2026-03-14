// Gollem agent with type-safe tools through LiteLLM.
//
// The tool parameters are Go structs — gollem generates the JSON schema
// automatically at compile time. LiteLLM passes tool definitions through
// transparently to the underlying provider.
//
// Usage:
//
//	litellm --model gpt-4o
//	go run ./tools
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/fugue-labs/gollem/core"
	"github.com/fugue-labs/gollem/provider/openai"
)

// WeatherParams defines the tool's input schema via struct tags.
// The JSON schema is generated at compile time — no runtime reflection needed.
type WeatherParams struct {
	City string `json:"city" description:"City name to get weather for"`
	Unit string `json:"unit,omitempty" description:"Temperature unit: celsius or fahrenheit"`
}

func main() {
	proxyURL := "http://localhost:4000"
	if u := os.Getenv("LITELLM_PROXY_URL"); u != "" {
		proxyURL = u
	}

	model := openai.NewLiteLLM(proxyURL,
		openai.WithModel("gpt-4o"),
	)

	// Define a type-safe tool. The function signature enforces correct types.
	weatherTool := core.FuncTool[WeatherParams](
		"get_weather",
		"Get current weather for a city",
		func(ctx context.Context, p WeatherParams) (string, error) {
			unit := p.Unit
			if unit == "" {
				unit = "fahrenheit"
			}
			// In production, call a real weather API here.
			return fmt.Sprintf("Weather in %s: 72°F (22°C), sunny", p.City), nil
		},
	)

	agent := core.NewAgent[string](model,
		core.WithTools[string](weatherTool),
		core.WithSystemPrompt[string]("You are a helpful weather assistant. Use the get_weather tool to answer weather questions."),
	)

	result, err := agent.Run(context.Background(), "What's the weather like in San Francisco and Tokyo?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}

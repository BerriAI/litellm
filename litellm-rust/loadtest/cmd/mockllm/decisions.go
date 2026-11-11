package main

import (
	"encoding/json"
	"fmt"
)

type decisionQuestion struct {
	Type    string `json:"type"`
	Name    string `json:"name"`
	Choices []struct {
		Value any `json:"value"`
	} `json:"choices"`
	Levels []struct {
		Label string `json:"label"`
	} `json:"levels"`
}

func decisionAnswers(body []byte) ([]any, error) {
	var request struct {
		Questions []decisionQuestion `json:"questions"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, err
	}
	if len(request.Questions) == 0 {
		return nil, fmt.Errorf("no questions")
	}
	answers := make([]any, 0, len(request.Questions))
	for _, q := range request.Questions {
		switch q.Type {
		case "predicate":
			answers = append(answers, map[string]any{"type": "predicate", "name": q.Name, "probability": 0.9})
		case "choice":
			if len(q.Choices) == 0 {
				return nil, fmt.Errorf("choice question %q has no choices", q.Name)
			}
			probabilities := make([]any, len(q.Choices))
			rest := 0.1 / float64(max(len(q.Choices)-1, 1))
			for i, c := range q.Choices {
				p := rest
				if i == 0 {
					p = 0.9
				}
				probabilities[i] = map[string]any{"value": c.Value, "probability": p}
			}
			answers = append(answers, map[string]any{"type": "choice", "name": q.Name, "choice": q.Choices[0].Value,
				"probabilities": probabilities, "confidence": 0.9})
		case "score":
			if len(q.Levels) == 0 {
				return nil, fmt.Errorf("score question %q has no levels", q.Name)
			}
			top := len(q.Levels) - 1
			probabilities := make([]any, len(q.Levels))
			rest := 0.1 / float64(max(top, 1))
			for i, l := range q.Levels {
				p := rest
				if i == top {
					p = 0.9
				}
				probabilities[i] = map[string]any{"value": i, "label": l.Label, "probability": p}
			}
			answers = append(answers, map[string]any{"type": "score", "name": q.Name, "score": top,
				"probabilities": probabilities, "confidence": 0.9})
		case "box":
			answers = append(answers, map[string]any{"type": "box", "name": q.Name,
				"box": map[string]any{"top": 0.1, "bottom": 0.9, "left": 0.1, "right": 0.9}, "confidence": 0.9})
		default:
			answers = append(answers, map[string]any{"type": "refusal", "name": q.Name})
		}
	}
	return answers, nil
}

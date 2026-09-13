package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

var managementPrefixes = map[string]bool{
	"access_group":       true,
	"agent":              true,
	"budget":             true,
	"cache":              true,
	"config":             true,
	"coordination_redis": true,
	"credentials":        true,
	"customer":           true,
	"fallback":           true,
	"guardrails":         true,
	"jwt":                true,
	"key":                true,
	"model":              true,
	"organization":       true,
	"project":            true,
	"prompts":            true,
	"router":             true,
	"search_tools":       true,
	"tag":                true,
	"team":               true,
	"user":               true,
	"vector_store":       true,
}

func isManagementPath(path string) bool {
	segments := strings.SplitN(strings.TrimPrefix(path, "/"), "/", 2)
	return len(segments) > 0 && managementPrefixes[segments[0]]
}

func parseAllowlist(path string) (map[string]bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	entries := make(map[string]bool)
	scanner := bufio.NewScanner(file)
	line := 0
	for scanner.Scan() {
		line++
		text := strings.TrimSpace(scanner.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		if idx := strings.Index(text, "#"); idx >= 0 {
			text = strings.TrimSpace(text[:idx])
		}
		fields := strings.Fields(text)
		if len(fields) != 2 || !strings.HasPrefix(fields[1], "/") {
			return nil, fmt.Errorf("%s:%d: allowlist entries must be \"METHOD /path\", got %q", path, line, text)
		}
		entries[strings.ToUpper(fields[0])+" "+fields[1]] = true
	}
	return entries, scanner.Err()
}

func specCallCovered(calls []endpointCall, specMethod, specPath string) bool {
	for _, call := range calls {
		if strings.EqualFold(call.Method, specMethod) && pathMatches(call.Path, specPath) {
			return true
		}
	}
	return false
}

func auditCoverage(calls []endpointCall, specPaths map[string]map[string]json.RawMessage, allowlist map[string]bool) []string {
	var violations []string
	seen := make(map[string]bool)
	for specPath, operations := range specPaths {
		if !isManagementPath(specPath) {
			continue
		}
		for method := range operations {
			entry := strings.ToUpper(method) + " " + specPath
			covered := specCallCovered(calls, method, specPath)
			switch {
			case allowlist[entry]:
				seen[entry] = true
				if covered {
					violations = append(violations, fmt.Sprintf("stale allowlist entry: %s is covered by the provider; remove it from the allowlist", entry))
				}
			case !covered:
				violations = append(violations, fmt.Sprintf("uncovered management endpoint: %s has no provider resource or data source; add coverage or allowlist it with a reason", entry))
			}
		}
	}
	for entry := range allowlist {
		if !seen[entry] {
			violations = append(violations, fmt.Sprintf("stale allowlist entry: %s is not a management endpoint in the proxy schema; remove it from the allowlist", entry))
		}
	}
	sort.Strings(violations)
	return violations
}

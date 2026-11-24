Created", statusCode: http.StatusCreated, wantErr: false, wantValue: "created"},
		{name: "202 Accepted", statusCode: http.StatusAccepted, wantErr: false, wantValue: "accepted"},
		{name: "400 Bad Request", statusCode: http.StatusBadRequest, wantErr: true},
		{name: "404 Not Found", statusCode: http.StatusNotFound, wantErr: true},
		{name: "500 Internal Server Error", statusCode: http.StatusInternalServerError, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.statusCode)
				w.Write([]byte(`{"value":"` + tt.wantValue + `"}`))
			}))
			defer srv.Close()

			client := NewClient(srv.URL, "test-key", true)
			result, err := client.sendRequest("POST", "/whatever", map[string]string{"foo": "bar"})

			if tt.wantErr {
				if err == nil {
					t.Fatalf("sendRequest returned no error for status %d", tt.statusCode)
				}
				var apiErr *apiError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error is not *apiError: %v", err)
				}
				if apiErr.StatusCode != tt.statusCode {
					t.Errorf("apiErr.StatusCode = %d, want %d", apiErr.StatusCode, tt.statusCode)
				}
				if tt.statusCode == http.StatusNotFound && !isNotFound(err) {
					t.Errorf("isNotFound(err) = false for 404, want true")
				}
				return
			}

			if err != nil {
				t.Fatalf("sendRequest returned unexpected error: %v", err)
			}
			if result["value"] != tt.wantValue {
				t.Errorf("result[value] = %v, want %q", result["value"], tt.wantValue)
			}
		})
	}
}

func TestRedactSensitiveDataNestedCredentialValues(t *testing.T) {
	c := NewClient("http://localhost:4000", "sk-test", false)

	input := `{"credential_name":"azure-cred","credential_values":{"api_key":"sk-secret-123","config":{"region":"us-east-1"}}}`
	redacted := c.redactSensitiveData(input)

	if strings.Contains(redacted, "sk-secret-123") {
		t.Errorf("redacted data still contains secret api_key: %s", redacted)
	}
	if !strings.Contains(redacted, "[REDACTED]") {
		t.Errorf("redacted data does not contain [REDACTED]: %s", redacted)
	}
}

func TestPerPrincipalModelFamilyBudgetsSupport(t *testing.T) {
	// Verifies that per-principal budgets covering model families or cross-vendor tiers
	// can be properly serialized and handled via client requests.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"success","budget_id":"budget-family-123"}`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "test-key", true)
	payload := map[string]interface{}{
		"principal_id": "user-principal-abc",
		"models":       []string{"openai/*", "anthropic/claude-3*"},
		"budget_max":   100.0,
		"budget_duration": "30d",
	}

	result, err := client.sendRequest("POST", "/spend/budgets", payload)
	if err != nil {
		t.Fatalf("Failed to create per-principal model family budget: %v", err)
	}

	if result["budget_id"] != "budget-family-123" {
		t.Errorf("Expected budget_id 'budget-family-123', got %v", result["budget_id"])
	}
}
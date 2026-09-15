package litellm

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func TestHandleAPIResponseAcceptsFullSuccessRange(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		wantErr    bool
	}{
		{name: "200 OK", statusCode: http.StatusOK, wantErr: false},
		{name: "201 Created", statusCode: http.StatusCreated, wantErr: false},
		{name: "202 Accepted", statusCode: http.StatusAccepted, wantErr: false},
		{name: "400 Bad Request", statusCode: http.StatusBadRequest, wantErr: true},
		{name: "404 Not Found", statusCode: http.StatusNotFound, wantErr: true},
		{name: "500 Internal Server Error", statusCode: http.StatusInternalServerError, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			rec.WriteHeader(tt.statusCode)
			rec.WriteString(`{"model_name":"gpt-4o"}`)
			resp := rec.Result()

			client := NewClient("http://localhost:4000", "test-key", true)
			got, err := handleAPIResponse(resp, map[string]interface{}{"model_name": "gpt-4o"}, client)

			if tt.wantErr {
				if err == nil {
					t.Fatalf("handleAPIResponse returned no error for status %d", tt.statusCode)
				}
				if !strings.Contains(err.Error(), strconv.Itoa(tt.statusCode)) {
					t.Errorf("error %q does not mention status %d", err.Error(), tt.statusCode)
				}
				return
			}

			if err != nil {
				t.Fatalf("handleAPIResponse returned unexpected error: %v", err)
			}
			if got.ModelName != "gpt-4o" {
				t.Errorf("got.ModelName = %q, want gpt-4o", got.ModelName)
			}
		})
	}
}

func TestHandleAPIResponseAcceptsEmptyBodyOn2xx(t *testing.T) {
	rec := httptest.NewRecorder()
	rec.WriteHeader(http.StatusNoContent)
	resp := rec.Result()

	client := NewClient("http://localhost:4000", "test-key", true)
	got, err := handleAPIResponse(resp, map[string]interface{}{"model_name": "gpt-4o"}, client)

	if err != nil {
		t.Fatalf("handleAPIResponse returned unexpected error for empty-body 204: %v", err)
	}
	if got == nil {
		t.Fatal("handleAPIResponse returned nil ModelResponse for empty-body 204")
	}
}

func TestHandleMCPAPIResponseAcceptsFullSuccessRange(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		wantErr    bool
	}{
		{name: "200 OK", statusCode: http.StatusOK, wantErr: false},
		{name: "201 Created", statusCode: http.StatusCreated, wantErr: false},
		{name: "202 Accepted", statusCode: http.StatusAccepted, wantErr: false},
		{name: "400 Bad Request", statusCode: http.StatusBadRequest, wantErr: true},
		{name: "404 Not Found", statusCode: http.StatusNotFound, wantErr: true},
		{name: "500 Internal Server Error", statusCode: http.StatusInternalServerError, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			rec.WriteHeader(tt.statusCode)
			rec.WriteString(`{"server_id":"srv-1","server_name":"gh"}`)
			resp := rec.Result()

			client := NewClient("http://localhost:4000", "test-key", true)
			var mcpResp MCPServerResponse
			err := handleMCPAPIResponse(resp, &mcpResp, client)

			if tt.wantErr {
				if err == nil {
					t.Fatalf("handleMCPAPIResponse returned no error for status %d", tt.statusCode)
				}
				if !strings.Contains(err.Error(), strconv.Itoa(tt.statusCode)) {
					t.Errorf("error %q does not mention status %d", err.Error(), tt.statusCode)
				}
				return
			}

			if err != nil {
				t.Fatalf("handleMCPAPIResponse returned unexpected error: %v", err)
			}
			if mcpResp.ServerID != "srv-1" {
				t.Errorf("mcpResp.ServerID = %q, want srv-1", mcpResp.ServerID)
			}
		})
	}
}

func TestHandleMCPAPIResponseRejectsEmptyBodyOn2xx(t *testing.T) {
	rec := httptest.NewRecorder()
	rec.WriteHeader(http.StatusNoContent)
	resp := rec.Result()

	client := NewClient("http://localhost:4000", "test-key", true)
	var mcpResp MCPServerResponse
	err := handleMCPAPIResponse(resp, &mcpResp, client)

	if err == nil {
		t.Fatal("handleMCPAPIResponse returned no error for empty-body 204; every MCP caller (create/read/update) writes the parsed result straight into Terraform state with no fallback, so a silently-accepted empty body would blank out or empty-ID the resource")
	}
}

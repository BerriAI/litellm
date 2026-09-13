package litellm

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestHandleMCPAPIResponse_Accepts201(t *testing.T) {
	body := `{"server_id":"test-123","server_name":"test","url":"https://example.com"}`
	resp := &http.Response{
		StatusCode: 201,
		Status:     "201 Created",
		Body:       io.NopCloser(strings.NewReader(body)),
	}

	var result MCPServerResponse
	err := handleMCPAPIResponse(resp, &result, &Client{})
	if err != nil {
		t.Fatalf("expected no error for 201 Created, got: %v", err)
	}
	if result.ServerID != "test-123" {
		t.Fatalf("expected server_id=test-123, got: %s", result.ServerID)
	}
}

func TestHandleMCPAPIResponse_Accepts202(t *testing.T) {
	body := `{"server_id":"del-456","server_name":"deleted"}`
	resp := &http.Response{
		StatusCode: 202,
		Status:     "202 Accepted",
		Body:       io.NopCloser(strings.NewReader(body)),
	}

	var result MCPServerResponse
	err := handleMCPAPIResponse(resp, &result, &Client{})
	if err != nil {
		t.Fatalf("expected no error for 202 Accepted, got: %v", err)
	}
	if result.ServerID != "del-456" {
		t.Fatalf("expected server_id=del-456, got: %s", result.ServerID)
	}
}

func TestHandleMCPAPIResponse_Accepts200(t *testing.T) {
	body := `{"server_id":"ok-789","server_name":"ok"}`
	resp := &http.Response{
		StatusCode: 200,
		Status:     "200 OK",
		Body:       io.NopCloser(strings.NewReader(body)),
	}

	var result MCPServerResponse
	err := handleMCPAPIResponse(resp, &result, &Client{})
	if err != nil {
		t.Fatalf("expected no error for 200 OK, got: %v", err)
	}
}

func TestHandleMCPAPIResponse_Rejects400(t *testing.T) {
	body := `{"error":{"message":"bad request"}}`
	resp := &http.Response{
		StatusCode: 400,
		Status:     "400 Bad Request",
		Body:       io.NopCloser(strings.NewReader(body)),
	}

	var result MCPServerResponse
	err := handleMCPAPIResponse(resp, &result, &Client{})
	if err == nil {
		t.Fatal("expected error for 400, got nil")
	}
}

func TestHandleMCPAPIResponse_Rejects500(t *testing.T) {
	body := `{"error":{"message":"internal error"}}`
	resp := &http.Response{
		StatusCode: 500,
		Status:     "500 Internal Server Error",
		Body:       io.NopCloser(strings.NewReader(body)),
	}

	var result MCPServerResponse
	err := handleMCPAPIResponse(resp, &result, &Client{})
	if err == nil {
		t.Fatal("expected error for 500, got nil")
	}
}

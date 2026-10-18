package main

import "testing"

func TestShortenKeepsFieldsAndNumbers(t *testing.T) {
	in := []byte(`{"messages":[{"content":"abcdefgh"}],"response_cost":0.10262000000000001,"n":3,"ok":true}`)
	got := string(shorten(in, 3))
	want := `{"messages":[{"content":"abc"}],"n":3,"ok":true,"response_cost":0.10262000000000001}`
	if got != want {
		t.Fatalf("got %s", got)
	}
	if got := string(shorten([]byte("not json"), 3)); got != "not json" {
		t.Fatalf("non-JSON changed: %s", got)
	}
}

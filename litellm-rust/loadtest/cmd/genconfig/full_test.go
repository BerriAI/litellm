package main

import (
	"strings"
	"testing"
)

func TestWithoutCallbacksDropsS3AndItsParams(t *testing.T) {
	got := withoutCallbacks(fullFeatureSettings, " s3_v2 ")
	for _, gone := range []string{"- s3_v2", "s3_callback_params", "s3_bucket_name"} {
		if strings.Contains(got, gone) {
			t.Errorf("still has %q:\n%s", gone, got)
		}
	}
	for _, kept := range []string{"- prometheus", "- dynamic_rate_limiter_v3", "  hooks:", "    - forward_headers:", "x-caller-team"} {
		if !strings.Contains(got, kept) {
			t.Errorf("lost %q:\n%s", kept, got)
		}
	}
	if withoutCallbacks(fullFeatureSettings, "") != fullFeatureSettings {
		t.Error("an empty list changed the settings")
	}
}

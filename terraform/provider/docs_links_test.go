package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The Terraform Registry serves these pages under /providers/BerriAI/litellm/<version>/docs
// without a trailing slash and does not rewrite relative markdown links, so a relative target
// resolves one segment too high and renders the registry's 404 page.
var markdownLink = regexp.MustCompile(`\[[^\]]*\]\(([^)]+)\)`)

func TestDocsLinksAreAbsolute(t *testing.T) {
	err := filepath.WalkDir("docs", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}

		content, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}

		for _, match := range markdownLink.FindAllStringSubmatch(string(content), -1) {
			target := strings.TrimSpace(match[1])
			if strings.HasPrefix(target, "#") || strings.Contains(target, "://") || strings.HasPrefix(target, "mailto:") {
				continue
			}
			t.Errorf("%s links to %q; registry docs need an absolute URL, e.g. "+
				"https://registry.terraform.io/providers/BerriAI/litellm/latest/docs/resources/team", path, target)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking docs: %v", err)
	}
}

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type endpointCall struct {
	Method string
	Path   string
	Pos    string
}

type extraction struct {
	Calls      []endpointCall
	Unresolved []string
}

var formatVerbPattern = regexp.MustCompile(`%[sdv]`)

func normalizePath(raw string) string {
	withoutQuery := strings.SplitN(raw, "?", 2)[0]
	return formatVerbPattern.ReplaceAllString(withoutQuery, "{param}")
}

func stringLit(expr ast.Expr) (string, bool) {
	lit, ok := expr.(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING {
		return "", false
	}
	value, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", false
	}
	return value, true
}

func packageConsts(files []*ast.File) map[string]string {
	consts := make(map[string]string)
	for _, file := range files {
		for _, decl := range file.Decls {
			genDecl, ok := decl.(*ast.GenDecl)
			if !ok || (genDecl.Tok != token.CONST && genDecl.Tok != token.VAR) {
				continue
			}
			for _, spec := range genDecl.Specs {
				valueSpec, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for i, name := range valueSpec.Names {
					if i >= len(valueSpec.Values) {
						continue
					}
					if value, ok := stringLit(valueSpec.Values[i]); ok {
						consts[name.Name] = value
					}
				}
			}
		}
	}
	return consts
}

func isSprintf(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "Sprintf" {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	return ok && pkg.Name == "fmt"
}

func resolveExpr(expr ast.Expr, fn *ast.FuncDecl, consts map[string]string) []string {
	switch node := expr.(type) {
	case *ast.BasicLit:
		if value, ok := stringLit(node); ok {
			return []string{value}
		}
	case *ast.Ident:
		if value, ok := consts[node.Name]; ok {
			return []string{value}
		}
		return resolveLocalIdent(node, fn, consts)
	case *ast.CallExpr:
		if isSprintf(node) && len(node.Args) > 0 {
			return resolveSprintf(node, fn, consts)
		}
	}
	return nil
}

func resolveSprintf(call *ast.CallExpr, fn *ast.FuncDecl, consts map[string]string) []string {
	formats := resolveExpr(call.Args[0], fn, consts)
	results := formats
	for _, arg := range call.Args[1:] {
		argValues := resolveExpr(arg, fn, consts)
		substituted := make([]string, 0, len(results))
		for _, format := range results {
			verb := formatVerbPattern.FindStringIndex(format)
			if verb == nil {
				substituted = append(substituted, format)
				continue
			}
			if len(argValues) == 0 {
				substituted = append(substituted, format[:verb[0]]+"\x00param\x00"+format[verb[1]:])
				continue
			}
			for _, argValue := range argValues {
				substituted = append(substituted, format[:verb[0]]+argValue+format[verb[1]:])
			}
		}
		results = substituted
	}
	restored := make([]string, 0, len(results))
	for _, result := range results {
		restored = append(restored, strings.ReplaceAll(result, "\x00param\x00", "%s"))
	}
	return restored
}

func resolveLocalIdent(ident *ast.Ident, fn *ast.FuncDecl, consts map[string]string) []string {
	if fn == nil {
		return nil
	}
	var values []string
	ast.Inspect(fn.Body, func(node ast.Node) bool {
		assign, ok := node.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for i, lhs := range assign.Lhs {
			lhsIdent, ok := lhs.(*ast.Ident)
			if !ok || lhsIdent.Name != ident.Name || i >= len(assign.Rhs) {
				continue
			}
			values = append(values, resolveExpr(assign.Rhs[i], fn, consts)...)
		}
		return true
	})
	return values
}

func requestCallMethodAndPath(call *ast.CallExpr) (methodArg ast.Expr, pathArg ast.Expr, matched bool) {
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		if fun.Sel.Name == "sendRequest" && len(call.Args) >= 2 {
			return call.Args[0], call.Args[1], true
		}
	case *ast.Ident:
		if fun.Name == "MakeRequest" && len(call.Args) >= 3 {
			return call.Args[1], call.Args[2], true
		}
	}
	return nil, nil, false
}

func isRawHTTPRequest(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || (sel.Sel.Name != "NewRequest" && sel.Sel.Name != "NewRequestWithContext") {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	return ok && pkg.Name == "http"
}

func extractFromFiles(fset *token.FileSet, files []*ast.File, helperFiles map[string]bool) extraction {
	consts := packageConsts(files)
	var result extraction
	for _, file := range files {
		fileName := fset.Position(file.Pos()).Filename
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			ast.Inspect(fn.Body, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok {
					return true
				}
				pos := fset.Position(call.Pos()).String()
				if isRawHTTPRequest(call) && !helperFiles[fileName] {
					result.Unresolved = append(result.Unresolved,
						fmt.Sprintf("%s: raw http.NewRequest outside the request helpers; route it through Client.sendRequest or MakeRequest", pos))
					return true
				}
				methodArg, pathArg, matched := requestCallMethodAndPath(call)
				if !matched {
					return true
				}
				methods := resolveExpr(methodArg, fn, consts)
				paths := resolveExpr(pathArg, fn, consts)
				if len(methods) == 0 || len(paths) == 0 {
					result.Unresolved = append(result.Unresolved,
						fmt.Sprintf("%s: cannot statically resolve method or path; use a string literal, package const, or fmt.Sprintf with a literal format", pos))
					return true
				}
				for _, method := range methods {
					for _, path := range paths {
						result.Calls = append(result.Calls, endpointCall{Method: method, Path: normalizePath(path), Pos: pos})
					}
				}
				return true
			})
		}
	}
	return result
}

func extractProviderCalls(providerDir string) (extraction, error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, providerDir, func(info os.FileInfo) bool {
		return !strings.HasSuffix(info.Name(), "_test.go")
	}, 0)
	if err != nil {
		return extraction{}, err
	}
	var files []*ast.File
	helperFiles := make(map[string]bool)
	for _, pkg := range pkgs {
		fileNames := make([]string, 0, len(pkg.Files))
		for name := range pkg.Files {
			fileNames = append(fileNames, name)
		}
		sort.Strings(fileNames)
		for _, name := range fileNames {
			files = append(files, pkg.Files[name])
			base := name[strings.LastIndex(name, "/")+1:]
			if base == "client.go" || base == "utils.go" {
				helperFiles[name] = true
			}
		}
	}
	return extractFromFiles(fset, files, helperFiles), nil
}

func loadSpecPaths(specPath string) (map[string]map[string]json.RawMessage, error) {
	data, err := os.ReadFile(specPath)
	if err != nil {
		return nil, err
	}
	var spec struct {
		Paths map[string]map[string]json.RawMessage `json:"paths"`
	}
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, err
	}
	if len(spec.Paths) == 0 {
		return nil, fmt.Errorf("spec %s contains no paths", specPath)
	}
	return spec.Paths, nil
}

func segmentsMatch(providerSegment, specSegment string) bool {
	if providerSegment == "{param}" {
		return strings.HasPrefix(specSegment, "{") && strings.HasSuffix(specSegment, "}")
	}
	return providerSegment == specSegment
}

func pathMatches(providerPath, specPath string) bool {
	providerSegments := strings.Split(strings.Trim(providerPath, "/"), "/")
	specSegments := strings.Split(strings.Trim(specPath, "/"), "/")
	if len(providerSegments) != len(specSegments) {
		return false
	}
	for i := range providerSegments {
		if !segmentsMatch(providerSegments[i], specSegments[i]) {
			return false
		}
	}
	return true
}

func auditCalls(calls []endpointCall, specPaths map[string]map[string]json.RawMessage) []string {
	var violations []string
	for _, call := range calls {
		pathFound := false
		methodFound := false
		for specPath, operations := range specPaths {
			if !pathMatches(call.Path, specPath) {
				continue
			}
			pathFound = true
			if _, ok := operations[strings.ToLower(call.Method)]; ok {
				methodFound = true
				break
			}
		}
		if !pathFound {
			violations = append(violations, fmt.Sprintf("%s: %s %s is not served by the proxy", call.Pos, call.Method, call.Path))
		} else if !methodFound {
			violations = append(violations, fmt.Sprintf("%s: %s %s: path exists but method not allowed", call.Pos, call.Method, call.Path))
		}
	}
	return violations
}

func run(providerDir, specPath string) error {
	extracted, err := extractProviderCalls(providerDir)
	if err != nil {
		return err
	}
	if len(extracted.Unresolved) > 0 {
		return fmt.Errorf("unresolved call sites:\n  %s", strings.Join(extracted.Unresolved, "\n  "))
	}
	if len(extracted.Calls) == 0 {
		return fmt.Errorf("extracted zero request call sites from %s; extractor or provider layout changed", providerDir)
	}
	specPaths, err := loadSpecPaths(specPath)
	if err != nil {
		return err
	}
	violations := auditCalls(extracted.Calls, specPaths)
	if len(violations) > 0 {
		sort.Strings(violations)
		return fmt.Errorf("provider/proxy endpoint drift:\n  %s", strings.Join(violations, "\n  "))
	}
	fmt.Printf("OK: %d request call sites verified against %d proxy OpenAPI paths\n", len(extracted.Calls), len(specPaths))
	return nil
}

func main() {
	providerDir := flag.String("provider-dir", "./litellm", "directory containing the provider Go source")
	specPath := flag.String("spec", "", "path to the proxy OpenAPI schema JSON")
	flag.Parse()
	if *specPath == "" {
		fmt.Fprintln(os.Stderr, "error: -spec is required")
		os.Exit(2)
	}
	if err := run(*providerDir, *specPath); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

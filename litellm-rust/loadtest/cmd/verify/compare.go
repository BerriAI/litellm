package main

import (
	"fmt"
	"math"
	"strings"
)

// compareSides checks that two sides, given identical requests, recorded identical spend, and
// lists the telemetry one side has and the other lacks. Missing telemetry is reported, not
// failed: some of it is deliberately left out of one gateway.
func compareSides(r *report, a, b string, da, db *dbTotals, ma, mb map[string]*family, la, lb *s3Totals) {
	both := a + " vs " + b
	if da != nil && db != nil {
		// Routing picks deployments at random and they are priced differently, so the sides'
		// spend agrees per deployment price, not in total.
		for _, name := range sortedKeys(da.Deployments) {
			pb := db.Deployments[name]
			if pb == nil {
				continue
			}
			inA, outA, okA := da.Deployments[name].prices()
			inB, outB, okB := pb.prices()
			if !okA || !okB {
				continue
			}
			price := func(in, out float64) string {
				return fmt.Sprintf("$%.4f / $%.4f per 1M", in*1e6, out*1e6)
			}
			r.add(check{Name: "input / output price for " + name, Side: both, Want: price(inA, outA),
				Got: price(inB, outB), OK: closeTo(inA, inB) && closeTo(outA, outB)})
		}
		r.add(info("total spend ("+a+" / "+b+")", both, fmt.Sprintf("$%.6f / $%.6f", da.Spend, db.Spend)))
	}
	if ma != nil && mb != nil {
		fa, fb := litellmFamilies(ma), litellmFamilies(mb)
		r.add(info("metrics only on "+a, both, missing(fa, fb)))
		r.add(info("metrics only on "+b, both, missing(fb, fa)))
		var labels []string
		for _, name := range sortedKeys(fa) {
			if f, ok := fb[name]; ok {
				if only := missing(fa[name].labels, f.labels); only != "" {
					labels = append(labels, name+": "+only)
				}
			}
		}
		r.add(info("labels only on "+a+" (shared metrics)", both, strings.Join(labels, "; ")))
	}
	if la != nil && lb != nil {
		r.add(info("S3 log fields only on "+a, both, missing(la.Fields, lb.Fields)))
		r.add(info("S3 log fields only on "+b, both, missing(lb.Fields, la.Fields)))
		r.add(info("S3 log metadata only on "+a, both, missing(la.Metadata, lb.Metadata)))
		r.add(info("S3 log metadata only on "+b, both, missing(lb.Metadata, la.Metadata)))
	}
}

func info(name, side, got string) check {
	if got == "" {
		got = "none"
	}
	return check{Name: name, Side: side, Want: "-", Got: got, OK: true, Note: "reported"}
}

// missing lists the keys of a that b lacks.
func missing[V any, W any](a map[string]V, b map[string]W) string {
	var out []string
	for _, k := range sortedKeys(a) {
		if _, ok := b[k]; !ok {
			out = append(out, k)
		}
	}
	return strings.Join(out, ", ")
}

// closeTo allows for the float error of fitting prices from summed spend.
func closeTo(a, b float64) bool {
	return math.Abs(a-b) <= 1e-4*math.Max(math.Abs(a), math.Abs(b))
}

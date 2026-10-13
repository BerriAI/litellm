import sys, json
G = {"PFX": "t1392"}
exec(open("/home/ubuntu/qa_ar/boot.py").read(), G)
exec(open("/home/ubuntu/qa_ar/lib.py").read(), G)
G["OUT"] = "/home/ubuntu/qa_ar/shots_1392"
exec(open("/home/ubuntu/qa_ar/cases.py").read(), G)
exec(open("/home/ubuntu/qa_ar/tip_extra.py").read(), G)
G["login"]()
for n in sys.argv[1:]: G["case"](n, G[n])
json.dump({"results": G["RESULTS"], "log": G["LOG"]}, open(f"/home/ubuntu/qa_ar/shots_1392/results_{'_'.join(sys.argv[1:])}.json", "w"), indent=2, default=str)

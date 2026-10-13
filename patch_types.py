import json, os, bluesky
P=os.path.join(os.path.dirname(bluesky.__file__),"resources/performance/OpenAP/rotor/aircraft.json")
def ensure_types():
    d=json.load(open(P)); ch=False
    new={
     "EH216":{"name":"EHang 216 (approx. public specs)","n_engines":16,"engine_type":"TS","mtow":600,"oew":360,"mfc":0,
              "engines":[[f"Motor-EH-{i+1}",15.0] for i in range(16)],
              "envelop":{"v_min":-36,"v_max":36,"vs_min":-5,"vs_max":5,"h_max":3000,"d_range_max":35}},
     "AAVM":{"name":"Generic medium eVTOL AAV (illustrative)","n_engines":8,"engine_type":"TS","mtow":1800,"oew":1100,"mfc":0,
              "engines":[[f"Motor-AAV-{i+1}",100.0] for i in range(8)],
              "envelop":{"v_min":-55,"v_max":55,"vs_min":-5,"vs_max":5,"h_max":3000,"d_range_max":80}}}
    for k,v in new.items():
        if k not in d: d[k]=v; ch=True
    if ch: json.dump(d,open(P,"w"),indent=1)
    return ch
if __name__=="__main__": print("patched" if ensure_types() else "already present")

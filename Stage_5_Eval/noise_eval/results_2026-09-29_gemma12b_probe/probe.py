import base64, json, sys, time, urllib.request
prompt = open(".scratch/gemma12b/prompt.txt", encoding="utf-8").read()
for wav in sys.argv[1:]:
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}],
        "temperature": 0, "top_k": 1, "max_tokens": 256, "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    req = urllib.request.Request("http://127.0.0.1:8189/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    json.dump(r, open(wav + ".json", "w", encoding="utf-8"), ensure_ascii=False)
    m = r["choices"][0]["message"]
    print(wav, f"{time.time()-t:.1f}s", r.get("timings", {}).get("prompt_n"), "prompt toks,",
          r.get("timings", {}).get("predicted_n"), "gen toks", flush=True)
    print("  content:", repr(m.get("content")))
    if m.get("reasoning_content"): print("  reasoning:", repr(m["reasoning_content"][:300]))

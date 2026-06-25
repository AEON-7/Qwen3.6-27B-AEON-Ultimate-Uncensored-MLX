"""Prove the prefix-cache win: multi-turn conversation with a long shared prefix,
measured with --enable-prefix-caching (APC on) vs without (off). Runs the PATCHED
git-main mlx-vlm (the clone) via PYTHONPATH. Reports per-turn request time +
prompt_tokens + cached_tokens. With APC, turn-2+ should reuse the prefix (cached>0)
and stay fast; without, it re-prefills the whole thing every turn."""
import subprocess, time, json, urllib.request, signal, os

VENV = "/Users/albert/gemma4-mlxfp4/.venv/bin/python"
MODEL = "/Users/albert/qwen36-mlx/out/mxfp4-mixed"
CLONE = "/Users/albert/mlx-vlm-src"
PORT = 8082

DOC = ("Analyze this technical note.\n\n" +
       "Mixed-precision quantization preserves accuracy by keeping sensitive "
       "residual-writer and SSM layers at higher precision while compressing the "
       "quant-tolerant bulk to 4-bit. " * 45)

def run_convo(apc_on):
    env = dict(os.environ); env["PYTHONPATH"] = CLONE
    cmd = [VENV, "-m", "mlx_vlm.server", "--model", MODEL, "--port", str(PORT), "--trust-remote-code"]
    if apc_on:
        cmd.append("--enable-prefix-caching")
    srv = subprocess.Popen(cmd, env=env, stdout=open("/tmp/apcsrv.log", "w"), stderr=subprocess.STDOUT)
    out = []
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2); break
            except Exception:
                time.sleep(2)
        msgs = [{"role": "user", "content": DOC + "\n\nQ1: summarize the key idea in one sentence."}]
        for turn in range(1, 5):
            body = {"model": MODEL, "messages": msgs, "max_tokens": 8, "temperature": 0.0}
            req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            t0 = time.time()
            resp = json.load(urllib.request.urlopen(req, timeout=240))
            dt = time.time() - t0
            u = resp.get("usage", {})
            det = u.get("prompt_tokens_details") or {}
            cached = (det.get("cached_tokens") if isinstance(det, dict) else None) or u.get("cached_tokens", 0)
            out.append((turn, dt, u.get("prompt_tokens", 0), cached))
            msgs.append({"role": "assistant", "content": resp["choices"][0]["message"]["content"]})
            msgs.append({"role": "user", "content": f"Q{turn+1}: elaborate briefly on that."})
    finally:
        srv.send_signal(signal.SIGINT); time.sleep(3)
        if srv.poll() is None:
            srv.terminate()
        time.sleep(3)
    return out

for apc_on in [False, True]:
    print(f"=== APC {'ON (--enable-prefix-caching)' if apc_on else 'OFF (default)'} ===", flush=True)
    for turn, dt, ptok, cached in run_convo(apc_on):
        print(f"  turn {turn}: {dt*1000:6.0f} ms | prompt_tokens {ptok:4d} | cached_tokens {cached:4d}", flush=True)
    time.sleep(4)
print("DONE", flush=True)

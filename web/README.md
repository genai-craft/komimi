# web/ — ブラウザデモ (WebAssembly)

`csrc/` のエンジンをそのまま Emscripten で WebAssembly にし、端末と同じ int8 演算 (`KM_ATT_INT8`) でブラウザ上で動かす。
S3 用 (8 層 sub_ch 88) と P4 用 (16 層) のモデルに同じ音を流して並べて比較できる。結果は端末と一致し、速さだけ PC の値。

```bash
web/build.sh                       # docker の emscripten/emsdk で web/dist/komimi.{js,wasm}
cp web/index.html web/worker.js web/kw_worker.js web/dist/ && cp <model>.kmm web/dist/models/
docker run -d --name komimi-web -p 172.20.0.1:8610:80 -v $PWD/web/dist:/usr/share/nginx/html:ro nginx:alpine
```

公開: komimi.aunvox.com (Aunvox の Cloudflare トンネルに ingress `komimi.aunvox.com → http://172.20.0.1:8610` を追加。
DNS の CNAME `komimi → <tunnel-id>.cfargotunnel.com` はダッシュボードで)。マイクは HTTPS が要る。
ワーカー 1 つ = エンジン 1 つ (UI スレッドを止めない)。ファイル入力は decodeAudioData → 16 kHz に線形補間。

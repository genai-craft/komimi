// kana-whisper (809M、sbintuitions/kana-whisper) をブラウザで。transformers.js (ONNX Runtime Web、WebGPU) を CDN から読む。
// ストリーミングではないので、UI 側が無音で区切った発話ごとに丸ごと渡す。
let pipe = null, loading = false;
self.onmessage = async (e) => {
  const d = e.data;
  if (d.type === 'init') {
    if (loading || pipe) return; loading = true;
    try {
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5');
      env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = d.base;   // 例 /models/
      const t0 = performance.now();
      // 量子化の精度 (dev 20 発話、教師の GPU 出力との CER): decoder fp32 1.1% / q4 4.3%、encoder fp16 1.1% / int8 4.9%。
      // 精度優先 = encoder fp16 + decoder fp32 (2.0 GB、GPU メモリ 3 GB 級が要る)。軽量 = encoder q4f16 + decoder q4 (0.7 GB)。
      const load = (device) => pipeline('automatic-speech-recognition', 'kana-whisper', {
        device, dtype: d.mode === 'full' ? { encoder_model: 'fp16', decoder_model_merged: 'fp32' } : { encoder_model: 'q4f16', decoder_model_merged: 'q4' },
        progress_callback: (p) => { postMessage({ type: 'status', status: p.status, file: p.file || '', pct: p.progress || 0 }); },
      });
      let device = d.device || 'webgpu';
      try { pipe = await load(device); }
      catch (err) {
        if (device !== 'webgpu') throw err;
        postMessage({ type: 'progress', file: 'WebGPU で読めず → CPU (wasm) で再試行: ' + String(err).slice(0, 80), pct: 0 });
        device = 'wasm'; pipe = await load(device);
      }
      postMessage({ type: 'ready', ms: performance.now() - t0, device });
    } catch (err) { postMessage({ type: 'error', msg: String(err) }); }
    loading = false;
  } else if (d.type === 'transcribe') {
    if (!pipe) return;
    const t0 = performance.now();
    try {
      const out = await pipe(d.pcm, { language: 'ja', task: 'transcribe', chunk_length_s: 30, return_timestamps: false });
      postMessage({ type: 'text', id: d.id, text: (out.text || '').trim(), ms: performance.now() - t0, sec: d.pcm.length / 16000 });
    } catch (err) { postMessage({ type: 'error', msg: String(err) }); }
  }
};

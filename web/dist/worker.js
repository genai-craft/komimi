// komimi の WebAssembly をワーカーで動かす (UI スレッドを止めない)。1 ワーカー = 1 エンジン。
importScripts('komimi.js');
let M = null, ctx = 0, feed = null, text = null, frames = null, skipped = null, busyUs = 0, audioS = 0, lastChunkMs = 0;
self.onmessage = async (e) => {
  const d = e.data;
  if (d.type === 'init') {
    M = await createKomimi();
    const buf = new Uint8Array(d.model);
    const p = M._malloc(buf.length); M.HEAPU8.set(buf, p);
    ctx = M.ccall('kmw_load', 'number', ['number', 'number'], [p, buf.length]); M._free(p);
    if (!ctx) { postMessage({ type: 'error', msg: 'model load failed' }); return; }
    const rc = M.ccall('kmw_stream_new', 'number', ['number', 'number', 'number', 'number'], [ctx, d.chunk, d.left, d.gateDb || 0]);
    M.ccall('kmw_set_dedup', null, ['number', 'number'], [ctx, d.dedup ? 1 : 0]);
    feed = M.cwrap('kmw_feed', 'number', ['number', 'number', 'number']);
    text = M.cwrap('kmw_text', 'string', ['number']); frames = M.cwrap('kmw_frames', 'number', ['number']); skipped = M.cwrap('kmw_skipped', 'number', ['number']);
    postMessage({ type: 'ready', layers: M.ccall('kmw_layers', 'number', ['number'], [ctx]), subCh: M.ccall('kmw_sub_ch', 'number', ['number'], [ctx]), rc });
  } else if (d.type === 'feed') {
    const pcm = d.pcm; const p = M._malloc(pcm.length * 4); M.HEAPF32.set(pcm, p >> 2);
    const t0 = performance.now(); const done = feed(ctx, p, pcm.length); const dt = performance.now() - t0; M._free(p);
    busyUs += dt * 1000; audioS += pcm.length / 16000;
    if (done > 0) lastChunkMs = dt / done;                                           // チャンク 1 つあたりの計算時間
    if (done > 0 || d.flush) postMessage({ type: 'text', text: text(ctx), frames: frames(ctx), skipped: skipped(ctx), rtf: audioS > 0 ? (busyUs / 1e6) / audioS : 0, chunkMs: lastChunkMs });
  } else if (d.type === 'finish') {
    M.ccall('kmw_finish', null, ['number'], [ctx]);
    postMessage({ type: 'text', text: text(ctx), frames: frames(ctx), skipped: skipped(ctx), rtf: audioS > 0 ? (busyUs / 1e6) / audioS : 0, final: true });
  } else if (d.type === 'reset') {
    M.ccall('kmw_reset', null, ['number'], [ctx]); busyUs = 0; audioS = 0;
    postMessage({ type: 'text', text: '', frames: 0, skipped: 0, rtf: 0 });
  }
};

/**
 * PLK AirCast — frontend v9 (เซิร์ฟเวอร์ FastAPI หรือ static บน GitHub Pages)
 * - กริด 1×1 กม. (รับเป็น array แบน แล้ววาดเป็นภาพ raster ใน MapLibre)
 * - ไทม์ไลน์รายชั่วโมง 2 โหมด: พยากรณ์ 0..+72 ชม. และย้อนหลัง −168..0 ชม. (7 วัน)
 * - โดมความสูงชั้นผสมอากาศ (deck.gl SolidPolygonLayer 3 มิติ) พร้อมสถานะตรวจสอบ
 * - เส้นลมแบบ Windy ผูกกับพิกัดจริง สีบอกความเร็ว
 */

// --------------------------------------------------------------------------
// ค่าคงที่
// --------------------------------------------------------------------------
const FC_STOPS = [0, 6, 12, 24, 48, 72];
const API_VERSION = 8;       // ต้องตรงกับ API_VERSION ใน app.py

// เกณฑ์ คพ. (2566) และสีตาม Air4Thai
const TIERS = [
  { level: 1, max: 15,   name: 'ดีมาก',            rgb: [59, 204, 255], css: 'var(--t1)', ink: 'var(--t1-ink)', hex: '#3bccff', range: '0–15',
    advice: 'อากาศดีมาก เหมาะกับกิจกรรมกลางแจ้งและการท่องเที่ยว' },
  { level: 2, max: 25,   name: 'ดี',               rgb: [146, 208, 80], css: 'var(--t2)', ink: 'var(--t2-ink)', hex: '#92d050', range: '15.1–25',
    advice: 'อากาศดี ทำกิจกรรมกลางแจ้งได้ตามปกติ' },
  { level: 3, max: 37.5, name: 'ปานกลาง',          rgb: [255, 255, 0],  css: 'var(--t3)', ink: 'var(--t3-ink)', hex: '#ffff00', range: '25.1–37.5',
    advice: 'ทำกิจกรรมได้ตามปกติ กลุ่มเสี่ยง (เด็ก ผู้สูงอายุ ผู้มีโรคทางเดินหายใจ) ควรสังเกตอาการ' },
  { level: 4, max: 75,   name: 'เริ่มมีผลต่อสุขภาพ', rgb: [255, 162, 0],  css: 'var(--t4)', ink: 'var(--t4-ink)', hex: '#ffa200', range: '37.6–75',
    advice: 'ลดกิจกรรมกลางแจ้งที่ใช้แรงมาก สวมหน้ากากกันฝุ่นเมื่ออยู่นอกอาคาร กลุ่มเสี่ยงควรอยู่ในอาคาร' },
  { level: 5, max: Infinity, name: 'มีผลต่อสุขภาพ', rgb: [240, 70, 70], css: 'var(--t5)', ink: 'var(--t5-ink)', hex: '#f04646', range: '> 75',
    advice: 'งดกิจกรรมกลางแจ้ง สวมหน้ากาก N95 เมื่อจำเป็นต้องออกนอกอาคาร หากมีอาการผิดปกติให้พบแพทย์' }
];
const tierOf = pm => TIERS.find(t => pm <= t.max) || TIERS[4];
const PREFERS_REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// --------------------------------------------------------------------------
// สถานะแอป
// --------------------------------------------------------------------------
const state = {
  mode: 'forecast',          // 'forecast' | 'history'
  offset: 0,                 // ชั่วโมงที่แสดงอยู่ (เทียบกับตอนนี้)
  target: 0,                 // ชั่วโมงที่ผู้ใช้เลือกล่าสุด (อาจกำลังโหลด)
  series: { forecast: null, history: null },
  grids: new Map(), surfaces: new Map(), domes: new Map(), inflight: new Map(),
  hotspots: null,
  selected: null,
  playing: false, playTimer: null,
  surf: { ready: false, active: 'A', shownKey: null },
  layers: { pmSurface: true, stations: true, wind: true, hotspots: true, volume: false }
};

// --------------------------------------------------------------------------
// โหมด static (GitHub Pages): ตอบคำขอ /api/... จาก data/bundle.json
// สูตรคำนวณเหมือน /api/grid ใน app.py ทุกขั้น (IDW ค่าฝุ่น, Gaussian kernel ลม/เพดานอากาศ, เกาะความร้อนเมือง)
// --------------------------------------------------------------------------
const StaticAPI = (() => {
  let B = null, HS = null, G = null, loading = null;
  const cache = new Map();

  function prepare() {
    const g = B.geo, nx = g.nx, ny = g.ny, n = nx * ny;
    const lat = new Float64Array(n), lon = new Float64Array(n);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { lat[j * nx + i] = g.g_lats[j]; lon[j * nx + i] = g.g_lons[i]; }
    const nw = g.wx_lat.length, ns = B.stations.length, c = g.cos_lat;
    // Gaussian kernel (25 จุดอากาศ -> ทุกช่องกริด) — kernel_weights()
    const K = new Float64Array(n * nw);
    for (let p = 0; p < n; p++) {
      let s = 0;
      for (let k = 0; k < nw; k++) {
        const dy = lat[p] - g.wx_lat[k], dx = (lon[p] - g.wx_lon[k]) * c;
        const w = Math.exp(-(dx * dx + dy * dy) / (2 * g.kernel_bw * g.kernel_bw));
        K[p * nw + k] = w; s += w;
      }
      s = Math.max(s, 1e-12);
      for (let k = 0; k < nw; k++) K[p * nw + k] /= s;
    }
    // IDW (สถานี -> ทุกช่องกริด) — idw_weights()
    const W = new Float64Array(n * ns);
    for (let p = 0; p < n; p++) {
      let s = 0;
      for (let k = 0; k < ns; k++) {
        const dy = lat[p] - B.stations[k].lat, dx = (lon[p] - B.stations[k].lon) * c;
        const w = 1 / (Math.pow(dx * dx + dy * dy, g.idw_power / 2) + 1e-4);
        W[p * ns + k] = w; s += w;
      }
      for (let k = 0; k < ns; k++) W[p * ns + k] /= s;
    }
    const uhi = new Float64Array(n);
    for (let p = 0; p < n; p++) {
      const dy = lat[p] - g.city[0], dx = (lon[p] - g.city[1]) * c;
      uhi[p] = Math.exp(-(dx * dx + dy * dy) / (2 * g.uhi_sigma * g.uhi_sigma));
    }
    G = { n, nw, ns, K, W, uhi };
  }

  // เวลาไทยแบบ "นาฬิกา" (ไม่ขึ้นกับเขตเวลาของเบราว์เซอร์)
  function shift(t0, off) {
    const [d, h] = t0.split(' ');
    const [Y, M, D] = d.split('-').map(Number);
    const t = new Date(Date.UTC(Y, M - 1, D, parseInt(h, 10)) + off * 3600e3).toISOString();
    return { day: t.slice(0, 10), time: `${t.slice(0, 10)} ${t.slice(11, 13)}:00` };
  }

  function snapshot(off) {
    const { day, time } = shift(B.t0, off);
    let rec = B.daily[day];
    if (!rec) {   // values_at(): ใช้วันที่ใกล้ที่สุด
      const days = Object.keys(B.daily).sort();
      rec = B.daily[days.reduce((a, b) => (Math.abs(new Date(b) - new Date(day)) < Math.abs(new Date(a) - new Date(day)) ? b : a))];
    }
    const vals = rec.values.slice(), src = rec.source.slice();
    const rt = off === 0 ? B.realtime : {};
    const stations = B.stations.map((s, i) => {
      const r = rt[s.point_id], model = rec.values[i];
      if (r) { vals[i] = r.value; src[i] = 'realtime'; }
      return { ...s, pm25: Math.round(vals[i] * 100) / 100, source: src[i], model_daily: Math.round(model * 100) / 100, realtime: r || null };
    });
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const r2 = x => Math.round(x * 100) / 100;
    return {
      offset: off, target_time: time, day,
      weather: B.weather[off - B.range[0]].city,
      provincial_avg: r2(mean), provincial_max: r2(Math.max(...vals)), provincial_min: r2(Math.min(...vals)),
      observed_share: Math.round(src.filter(x => x === 'obs' || x === 'realtime').length / src.length * 100) / 100,
      realtime_count: src.filter(x => x === 'realtime').length,
      stations
    };
  }

  function grid(off) {
    if (cache.has(off)) return cache.get(off);
    const snap = snapshot(off), wx = B.weather[off - B.range[0]];
    const { n, nw, ns, K, W, uhi } = G;
    const preds = snap.stations.map(s => s.pm25);
    const pm = new Array(n), mh = new Array(n), u = new Array(n), v = new Array(n);
    const cw = snap.weather;
    const td = Math.max(0.4, cw.surface_temp - cw.temp + 1.5);
    const lift = Math.min(0.25, 0.04 * td);
    for (let p = 0; p < n; p++) {
      let a = 0; for (let k = 0; k < ns; k++) a += W[p * ns + k] * preds[k];
      let su = 0, sv = 0, sm = 0;
      for (let k = 0; k < nw; k++) { const w = K[p * nw + k]; su += w * wx.u[k]; sv += w * wx.v[k]; sm += w * wx.mh[k]; }
      pm[p] = Math.round(a * 10) / 10;
      mh[p] = Math.round(sm * (1 + lift * uhi[p]));
      u[p] = Math.round(su * 100) / 100; v[p] = Math.round(sv * 100) / 100;
    }
    const wUp = Math.round(Math.sqrt(Math.max(0.05, 9.81 * (td / 300) * (cw.mixing_height / 120))) * 100) / 100;
    const g = {
      offset: off, time: snap.target_time, weather_source: B.weather_source,
      grid: { nx: B.geo.nx, ny: B.geo.ny, km: B.geo.km, bounds: B.geo.bounds },
      pm25: pm, mixing_height: mh, u, v,
      urban_lift: { center: [B.geo.city[1], B.geo.city[0]], updraft_velocity_ms: wUp, mixing_height: cw.mixing_height,
        temperature_delta: Math.round(td * 10) / 10,
        state: wUp >= 1.4 ? 'ยกตัวสูงมาก (ระบายฝุ่นเร็ว)' : wUp >= 0.75 ? 'ยกตัวปานกลาง' : 'อากาศปิดกดทับ (กักฝุ่นใกล้พื้น)' },
      snapshot: snap
    };
    if (cache.size > 60) cache.delete(cache.keys().next().value);
    cache.set(off, g);
    return g;
  }

  async function load() {
    if (B) return;
    if (!loading) {
      loading = (async () => {
        const [b, h] = await Promise.all([
          fetch(`data/bundle.json?t=${Date.now()}`, { cache: 'no-store' }).then(r => { if (!r.ok) throw new Error(`data/bundle.json → HTTP ${r.status}`); return r.json(); }),
          fetch(`data/hotspots.json?t=${Date.now()}`, { cache: 'no-store' }).then(r => (r.ok ? r.json() : { status: 'error', hotspots: [] }))
        ]);
        B = b; HS = h; prepare();
      })();
    }
    await loading;
  }

  async function get(url) {
    await load();
    const path = url.split('?')[0];
    if (path === '/api/forecast') return B.forecast;
    if (path === '/api/history') return B.history;
    if (path === '/api/hotspots') return HS;
    if (path === '/api/grid') {
      const off = parseInt(new URLSearchParams(url.split('?')[1] || '').get('offset') || '0', 10);
      return grid(Math.max(B.range[0], Math.min(B.range[1], off)));
    }
    throw new Error(`ไม่รองรับ ${url} ในโหมด static`);
  }
  return { get, bundle: () => B };
})();

const S = () => state.series[state.mode];
const $ = id => document.getElementById(id);
let map, overlay, stationChart;

window.addEventListener('error', e => showToast(`ข้อผิดพลาดในหน้าเว็บ: ${e.message} (${(e.filename || '').split('/').pop()}:${e.lineno})`, 0));
window.addEventListener('unhandledrejection', e => console.warn('unhandled', e.reason));

document.addEventListener('DOMContentLoaded', () => {
  buildLegend();
  initMap();
  Wind.init();
  bindUI();
  loadInitial();
});

// --------------------------------------------------------------------------
// แผนที่
// --------------------------------------------------------------------------
function initMap() {
  map = new maplibregl.Map({
    container: 'map',
    style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    center: [100.42, 16.86], zoom: 8.9, pitch: 0, bearing: 0,
    attributionControl: { compact: true }, maxPitch: 70
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'bottom-right');
  overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
  map.addControl(overlay);
  map.once('load', () => { fitProvince(); initSurfaceLayers(); });
  map.on('mousemove', onMapHoverSurface);
  map.on('mouseout', hideTip);
  window.addEventListener('resize', debounce(fitProvince, 300));
}

function fitProvince() {
  const wide = window.innerWidth > 820;
  map.fitBounds([[99.92, 16.30], [100.90, 17.42]], {
    padding: wide ? { top: 90, bottom: 160, left: 380, right: 300 } : { top: 70, bottom: 320, left: 20, right: 20 },
    duration: 0
  });
}

// --------------------------------------------------------------------------
// โหลดข้อมูล
// --------------------------------------------------------------------------
async function getJSON(url) {
  if (window.PLK_STATIC) return StaticAPI.get(url);   // GitHub Pages: ไม่มีเซิร์ฟเวอร์ ใช้ data/bundle.json
  let r;
  try {
    r = await fetch(url, { cache: 'no-store' });
  } catch (e) {
    throw new Error(`ติดต่อเซิร์ฟเวอร์ไม่ได้ (${url}) — ตรวจว่ารัน python app.py อยู่ และเปิดผ่าน http://localhost:8000`);
  }
  if (!r.ok) {
    let detail = '';
    try { const j = await r.json(); detail = j.error ? `${j.error}${j.trace ? ' @ ' + j.trace[j.trace.length - 2] : ''}` : JSON.stringify(j); }
    catch (_) { detail = (await r.text().catch(() => '')).slice(0, 200); }
    throw new Error(`${url} → HTTP ${r.status}${detail ? ': ' + detail : ''}`);
  }
  return r.json();
}

async function loadInitial() {
  let step = 'โหลดผลพยากรณ์ (/api/forecast)';
  try {
    $('loaderText').textContent = 'กำลังรันโมเดลพยากรณ์…';
    const [fc, hs] = await Promise.all([
      getJSON('/api/forecast'),
      getJSON('/api/hotspots').catch(e => { console.warn(e); return { status: 'error', hotspots: [] }; })
    ]);
    if (fc.api_version !== API_VERSION) {
      throw new Error(`เซิร์ฟเวอร์ที่รันอยู่เป็นเวอร์ชันเก่า (API ${fc.api_version ?? '≤4'} แต่หน้าเว็บต้องการ ${API_VERSION}) — `
        + 'ปิดหน้าต่าง Terminal ที่รัน python app.py อยู่ (Ctrl+C) แล้วรันใหม่ จากนั้นรีเฟรชหน้านี้');
    }
    state.series.forecast = fc;
    state.hotspots = hs;
    step = 'โหลดกริด 1 กม. (/api/grid)';
    await loadGrid(0);
    step = 'แสดงแถบสถานะ'; renderStatus();
    step = 'วาดไทม์ไลน์'; renderTimeline();
    step = 'แสดงผลบนแผนที่'; applyOffset(0);
    $('loader').classList.add('hidden');
    FC_STOPS.slice(1).forEach(o => loadGrid(o).catch(() => {}));   // เตรียมช่วงเวลาหลักไว้ก่อน
    const probs = fc.meta?.problems || [];
    if (probs.length) showToast('ระบบทำงานได้แต่มีปัญหาบางส่วน: ' + probs.join(' | '), 12000);
  } catch (err) {
    console.error(err);
    $('loaderText').textContent = 'โหลดข้อมูลไม่สำเร็จ';
    showToast(`โหลดข้อมูลไม่สำเร็จ ขั้นตอน: ${step} — ${err.message}`, 0);
  }
}

// โหลดกริด 1 กม. ของชั่วโมงใดชั่วโมงหนึ่ง (ไม่ซ้ำซ้อน, เก็บ cache)
function loadGrid(off) {
  if (state.grids.has(off)) return Promise.resolve(state.grids.get(off));
  if (state.inflight.has(off)) return state.inflight.get(off);
  const p = getJSON(`/api/grid?offset=${off}`).then(g => {
    state.grids.set(off, g);
    state.surfaces.set(off, buildSurfaceCanvas(g).toDataURL('image/png'));
    state.inflight.delete(off);
    if (state.grids.size > 120) {                       // จำกัดหน่วยความจำ
      const first = state.grids.keys().next().value;
      if (first !== state.offset) { state.grids.delete(first); state.surfaces.delete(first); state.domes.delete(first); }
    }
    return g;
  }).catch(e => { state.inflight.delete(off); throw e; });
  state.inflight.set(off, p);
  return p;
}

// เลือกชั่วโมง: ถ้ามีข้อมูลแล้วแสดงทันที ไม่งั้นแสดงสถานะโหลดและคงภาพเดิมไว้
function setOffset(off) {
  const [r0, r1] = S().range;
  off = Math.max(r0, Math.min(r1, Math.round(off)));
  state.target = off;
  renderTimelineCursor(off);
  if (state.grids.has(off)) { applyOffset(off); return; }
  $('tlLoading').hidden = false;
  loadGrid(off).then(() => { if (state.target === off) applyOffset(off); })
    .catch(err => { console.error(err); showToast('โหลดข้อมูลชั่วโมงนี้ไม่สำเร็จ'); })
    .finally(() => { if (!state.inflight.size) $('tlLoading').hidden = true; });
}

function applyOffset(off) {
  state.offset = off;
  $('tlLoading').hidden = !state.inflight.size || state.target === off;
  renderAll();
}

async function setMode(mode) {
  if (mode === state.mode) return;
  stopPlay();
  if (mode === 'history' && !state.series.history) {
    $('tlLoading').hidden = false;
    $('modeHistory').disabled = true;
    try {
      state.series.history = await getJSON('/api/history');
    } catch (e) {
      console.error(e);
      showToast('โหลดข้อมูลย้อนหลังไม่สำเร็จ');
      $('modeHistory').disabled = false;
      $('tlLoading').hidden = true;
      return;
    }
    $('modeHistory').disabled = false;
  }
  state.mode = mode;
  ['modeForecast', 'modeHistory'].forEach(id => {
    const on = (id === 'modeHistory') === (mode === 'history');
    $(id).classList.toggle('active', on);
    $(id).setAttribute('aria-pressed', on);
  });
  document.querySelector('.timeline').classList.toggle('history', mode === 'history');
  $('histNote').hidden = mode !== 'history';
  renderTimeline();
  if (mode === 'history') {
    stops().forEach(s => loadGrid(s.offset).catch(() => {}));
    setOffset(-24);
  } else {
    setOffset(0);
  }
}

// --------------------------------------------------------------------------
// แถบสถานะ
// --------------------------------------------------------------------------
function renderStatus() {
  const meta = state.series.forecast.meta || {};
  const chipData = $('chipData');
  if (meta.data_as_of) {
    const d = new Date(meta.data_as_of + 'T00:00:00');
    const days = Math.floor((Date.now() - d) / 86400000);
    chipData.querySelector('.chip-text').textContent = `ค่าตรวจวัดล่าสุด ${thDate(d)}`;
    chipData.classList.toggle('warn', days > 2);
    chipData.classList.toggle('ok', days <= 2);
    chipData.title = `โมเดลเดินหน้าทีละวันจากค่าตรวจวัดล่าสุด (${meta.data_as_of}) โดยใช้ค่าที่ตัวเองทำนายเป็นค่าย้อนหลังของวันถัดไป`
      + (days > 2 ? ` — ห่างจากวันนี้ ${days} วัน ความคลาดเคลื่อนสะสมได้` : '')
      + (meta.stations_stale ? `\n${meta.stations_stale} สถานีไม่มีค่าตรวจวัดล่าสุด ใช้ค่าเฉลี่ยจังหวัดเป็นจุดเริ่มต้นแทน` : '');
  }
  const chipW = $('chipWeather');
  const live = meta.weather_source === 'open-meteo';
  chipW.querySelector('.chip-text').textContent = live ? 'อากาศ: Open-Meteo' : 'อากาศ: ค่าจำลอง (ออฟไลน์)';
  chipW.classList.toggle('ok', live && !(meta.weather_missing || []).length);
  chipW.classList.toggle('warn', !live || (meta.weather_missing || []).length > 0);
  if ((meta.weather_missing || []).length) chipW.title = 'Open-Meteo ไม่มีตัวแปร: ' + meta.weather_missing.join(', ') + ' (ใช้ค่าเริ่มต้นแทน)';

  const chipH = $('chipHotspot');
  chipH.classList.remove('chip-muted', 'ok', 'warn');
  if (meta.hotspot_source === 'firms' && meta.hotspot_summary) {
    const sensor = meta.hotspot_sensor || '';
    chipH.querySelector('.chip-text').textContent = `จุดความร้อนวันนี้: ${meta.hotspot_summary.province_today} จุด (${sensor})`;
    chipH.classList.add('ok');
    chipH.title = 'นับตามวันปฏิทินเวลาไทย เฉพาะจุดในจังหวัดพิษณุโลก'
      + (sensor.includes('VIIRS') ? '\nหมายเหตุ: VIIRS ละเอียดกว่า MODIS ที่ใช้ฝึกโมเดล จำนวนจุดจึงสูงกว่าที่โมเดลเคยเห็น' : '');
    $('hotspotLayerLbl').textContent = `${sensor} ย้อนหลัง 5 วันจากเวลาที่เลือก`;
  } else {
    const msg = meta.hotspot_source === 'no_key' ? 'ยังไม่ได้ตั้งค่า key' : 'ดึงข้อมูลไม่สำเร็จ';
    chipH.querySelector('.chip-text').textContent = `จุดความร้อน: ${msg}`;
    chipH.classList.add('warn');
    chipH.title = (meta.hotspot_message || '') + ' — โมเดลใช้ค่าจุดความร้อนคงที่แทน';
    $('hotspotLayerLbl').textContent = msg;
  }
  const rt = meta.realtime;
  const chipR = $('chipRealtime');
  if (rt) {
    const okSrc = (rt.sources || []).filter(x => x.ok);
    chipR.querySelector('.chip-text').textContent = `เรียลไทม์: ${rt.stations_live}/${rt.stations_total} สถานี`;
    chipR.classList.toggle('ok', rt.stations_live > 0);
    chipR.classList.toggle('warn', rt.stations_live === 0);
    chipR.title = (rt.sources || []).map(x => `${x.name}: ${x.ok ? x.stations + ' สถานี' : 'ไม่สำเร็จ — ' + x.message}`).join('\n')
      + `\nดึงข้อมูลทุก ${rt.poll_minutes} นาที · ล่าสุด ${rt.last_poll || '-'} · สะสมแล้ว ${rt.log_rows} ค่า`
      + '\nค่าที่สะสมครบวันจะถูกใช้เป็นค่าตรวจวัดรายวันให้โมเดลอัตโนมัติ'
      + (okSrc.length ? '' : '\nตั้งค่าแหล่งข้อมูลได้ที่ไฟล์ realtime_sources.json');
  }
  if (meta.grid) $('gridLbl').textContent = `กริด ${meta.grid.km}×${meta.grid.km} กม. (${meta.grid.nx}×${meta.grid.ny} ช่อง) · IDW จากสถานี`;
  if (meta.model_loaded === false) showToast('ไม่พบไฟล์โมเดล — ระบบแสดงค่าฝุ่นล่าสุดของแต่ละสถานีแทนการพยากรณ์', 8000);
  if (meta.station_count) $('stationCountLbl').textContent = `${meta.station_count} สถานี · ค่าจากโมเดลรายจุด`;
}

// --------------------------------------------------------------------------
// เรนเดอร์
// --------------------------------------------------------------------------
function current() {
  const grid = state.grids.get(state.offset);
  return { off: state.offset, grid, snap: grid.snapshot };
}

function renderAll() {
  if (!state.series.forecast || !state.grids.has(state.offset)) return;
  renderHero();
  renderWhy();
  renderDistricts();
  renderTimelineCursor(state.offset);
  updateLayers();
  Wind.setField(current().grid);
  if (state.selected) renderDrawer();
}

function relLabel(off) {
  if (off === 0) return 'ตอนนี้';
  const a = Math.abs(off), d = Math.floor(a / 24), h = a % 24;
  const txt = (d ? `${d} วัน ` : '') + (h ? `${h} ชม.` : '');
  return off > 0 ? `อีก ${txt.trim()}` : `${txt.trim()}ที่แล้ว`;
}

function renderHero() {
  const { off, snap } = current();
  const tier = tierOf(snap.provincial_avg);
  const hero = $('hero');
  hero.style.setProperty('--tier', tier.css);
  hero.style.setProperty('--tier-ink', tier.ink);
  $('heroWhen').textContent = (off === 0 ? `ตอนนี้ · ข้อมูล ${thTime(parseLocal(snap.target_time))}` : `${relLabel(off)} · ${thDateTime(snap.target_time)}`)
    + ` · ${snap.realtime_count ? `เรียลไทม์ ${snap.realtime_count} สถานี ที่เหลือจากโมเดล`
        : snap.observed_share >= 1 ? 'ค่าตรวจวัด' : snap.observed_share > 0 ? `ตรวจวัด ${Math.round(snap.observed_share * 100)}% ที่เหลือจากโมเดล` : 'ค่าจากโมเดล'}`;
  tweenNumber($('heroValue'), snap.provincial_avg, 400);
  $('heroBadge').textContent = tier.name;
  $('heroAdvice').textContent = tier.advice;

  const st = snap.stations;
  const maxS = st.reduce((a, b) => (b.pm25 > a.pm25 ? b : a), st[0]);
  const minS = st.reduce((a, b) => (b.pm25 < a.pm25 ? b : a), st[0]);
  $('statMax').textContent = fmt(maxS.pm25); $('statMax').style.color = tierOf(maxS.pm25).ink;
  $('statMaxWhere').textContent = maxS.district;
  $('statMin').textContent = fmt(minS.pm25); $('statMin').style.color = tierOf(minS.pm25).ink;
  $('statMinWhere').textContent = minS.district;
  const exceed = st.filter(s => s.pm25 > 37.5).length;
  $('statExceed').textContent = exceed;
  $('statExceed').style.color = exceed ? 'var(--t4-ink)' : 'var(--ink)';
  $('statExceedOf').textContent = `จาก ${st.length} สถานี`;
}

function renderWhy() {
  const { grid, snap } = current();
  const w = snap.weather;
  $('windArrow').style.transform = `rotate(${(w.wind_dir + 180) % 360}deg)`;   // ชี้ทิศที่ลมพัดไป
  $('wxWind').textContent = `${w.wind_speed.toFixed(1)} m/s`;
  const windWord = w.wind_speed < 1 ? 'ลมสงบ' : w.wind_speed < 2.5 ? 'ลมอ่อน' : w.wind_speed < 5 ? 'ลมปานกลาง' : 'ลมแรง';
  $('wxWindNote').textContent = `${windWord} จากทิศ${compassTH(w.wind_dir)}`;
  const mh = Math.round(w.mixing_height);
  $('wxMix').textContent = `${mh.toLocaleString()} m`;
  $('wxMixNote').textContent = mh < 250 ? 'เพดานต่ำ ฝุ่นถูกกดใกล้พื้น' : mh < 600 ? 'ปานกลาง' : 'เพดานสูง ฝุ่นกระจายตัวได้ดี';

  const vent = w.wind_speed * w.mixing_height;
  $('ventValue').textContent = `${Math.round(vent).toLocaleString()} m²/s`;
  const pos = vent <= 350 ? (Math.max(vent, 50) - 50) / 300 / 3
            : vent <= 800 ? 1 / 3 + (vent - 350) / 450 / 3
            : 2 / 3 + Math.min(1, Math.log(vent / 800) / Math.log(6000 / 800)) / 3;
  $('ventMarker').style.left = `${(pos * 100).toFixed(1)}%`;

  // จุดความร้อน: นับจากเวลาที่เลือก (ใช้ได้ทั้งโหมดพยากรณ์และย้อนหลัง)
  const hs = hotspotsAt();
  const hsRow = $('hsRow');
  if (hs) {
    hsRow.hidden = false;
    $('hsCount').textContent = `${hs.daily[0]} จุด`;
    const days = hs.daily.slice().reverse();
    const mx = Math.max(1, ...days);
    $('hsBars').innerHTML = days.map((v, i) =>
      `<i style="height:${Math.max(2, (v / mx) * 22)}px" title="${i === days.length - 1 ? '24 ชม.ก่อนเวลาที่เลือก' : (days.length - 1 - i) + ' วันก่อนหน้า'}: ${v} จุด"></i>`).join('');
  } else hsRow.hidden = true;

  const u = grid.urban_lift;
  if (u) {
    $('urbanUpdraft').textContent = `${u.updraft_velocity_ms.toFixed(2)} m/s`;
    $('urbanState').textContent = `อ.เมืองพิษณุโลก — ${u.state}`;
  }
}

function renderDistricts() {
  const { snap } = current();
  const groups = {};
  snap.stations.forEach(s => { (groups[s.district] = groups[s.district] || []).push(s); });
  const list = Object.entries(groups).map(([d, arr]) => ({
    district: d, avg: arr.reduce((a, s) => a + s.pm25, 0) / arr.length, n: arr.length
  })).sort((a, b) => b.avg - a.avg);
  const maxVal = Math.max(50, ...list.map(d => d.avg));
  const ol = $('districtList');
  const existing = new Map([...ol.children].map(li => [li.dataset.d, li]));
  list.forEach((d, i) => {
    let li = existing.get(d.district);
    if (!li) {
      li = document.createElement('li');
      li.dataset.d = d.district;
      li.innerHTML = `<button><span class="d-name"></span><span class="d-bar"><i></i></span><span class="d-val"></span></button>`;
      li.querySelector('button').addEventListener('click', () => focusDistrict(li.dataset.d));
    }
    const t = tierOf(d.avg);
    li.querySelector('.d-name').textContent = d.district.replace(/^(อ\.|อำเภอ)\s*/, '');
    li.querySelector('.d-bar i').style.width = `${(d.avg / maxVal) * 100}%`;
    li.querySelector('.d-bar i').style.background = t.css;
    const v = li.querySelector('.d-val');
    v.textContent = fmt(d.avg); v.style.color = t.ink;
    li.querySelector('button').setAttribute('aria-label', `${d.district} ค่าเฉลี่ย ${fmt(d.avg)} ไมโครกรัม ระดับ${t.name} ${d.n} สถานี`);
    if (ol.children[i] !== li) ol.insertBefore(li, ol.children[i] || null);
  });
}

function focusDistrict(district) {
  const arr = current().snap.stations.filter(s => s.district === district);
  if (!arr.length) return;
  const lons = arr.map(s => s.lon), lats = arr.map(s => s.lat);
  const pad = window.innerWidth > 820 ? { top: 120, bottom: 180, left: 400, right: 420 } : { top: 80, bottom: 340, left: 30, right: 30 };
  map.fitBounds([[Math.min(...lons) - 0.04, Math.min(...lats) - 0.04], [Math.max(...lons) + 0.04, Math.max(...lats) + 0.04]],
    { padding: pad, maxZoom: 11.5, duration: PREFERS_REDUCED ? 0 : 1200 });
  openDrawer(arr.reduce((a, b) => (b.pm25 > a.pm25 ? b : a)).point_id);
}

// --------------------------------------------------------------------------
// ไทม์ไลน์ (ใช้ร่วมกันทั้ง 2 โหมด)
// --------------------------------------------------------------------------
function stops() {
  const s = S();
  if (state.mode === 'forecast') {
    return FC_STOPS.map(o => {
      const tr = s.hourly_trend.find(t => t.offset === o);
      return { offset: o, label: o === 0 ? 'ตอนนี้' : `+${o} ชม.`, sub: tr ? thShortDT(tr.iso) : '' };
    });
  }
  // ย้อนหลัง: เที่ยงวันของแต่ละวัน + ตอนนี้
  const out = s.hourly_trend.filter(t => t.iso.endsWith('T12:00') && t.offset < 0).map(t => {
    const d = new Date(t.iso);
    return { offset: t.offset, label: d.toLocaleDateString('th-TH', { weekday: 'short', day: 'numeric' }), sub: '12:00' };
  });
  out.push({ offset: 0, label: 'ตอนนี้', sub: '' });
  return out;
}

function renderTimeline() {
  const s = S();
  const trend = s.hourly_trend;
  const [r0, r1] = s.range;
  const span = r1 - r0;
  const svg = $('tlSvg');
  const W = 1000, H = 100;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const yMax = Math.max(40, Math.ceil(Math.max(...trend.map(t => t.pm25)) / 10) * 10 + 5);
  const x = o => ((o - r0) / span) * W;
  const y = v => H - (v / yMax) * (H - 6);

  let bands = '', lo = 0;
  TIERS.forEach(t => {
    const hi = Math.min(t.max, yMax);
    if (lo < yMax) bands += `<rect x="0" width="${W}" y="${y(hi)}" height="${y(lo) - y(hi)}" fill="${t.hex}" opacity=".16"/>`;
    lo = t.max;
  });
  let days = '';
  const dayLabels = [];
  trend.forEach(t => {
    if (t.offset > r0 && t.iso.endsWith('T00:00')) {
      days += `<line x1="${x(t.offset)}" x2="${x(t.offset)}" y1="0" y2="${H}" stroke="#7a8898" stroke-dasharray="2 3" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
      const d = new Date(t.iso);
      dayLabels.push(`<span class="tl-day" style="left:${((t.offset - r0) / span) * 100}%">${d.toLocaleDateString('th-TH', { weekday: 'short', day: 'numeric', month: 'short' })}</span>`);
    }
  });
  $('tlGraph').querySelectorAll('.tl-day').forEach(n => n.remove());
  $('tlGraph').insertAdjacentHTML('beforeend', dayLabels.join(''));

  const pts = trend.map(t => `${x(t.offset).toFixed(1)},${y(t.pm25).toFixed(1)}`);
  const area = `M${x(r0)},${H} L${pts.join(' L')} L${x(trend[trend.length - 1].offset)},${H} Z`;
  const stopsGrad = trend.map(t => `<stop offset="${((t.offset - r0) / span) * 100}%" stop-color="${darken(tierOf(t.pm25).rgb)}"/>`).join('');
  svg.innerHTML = `
    <defs><linearGradient id="tlLine" gradientUnits="userSpaceOnUse" x1="0" x2="${W}" y1="0" y2="0">${stopsGrad}</linearGradient></defs>
    ${bands}
    <path d="${area}" fill="#1b2838" opacity=".05"/>
    <polyline points="${pts.join(' ')}" fill="none" stroke="url(#tlLine)" stroke-width="2.5" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>
    ${days}`;

  const wrap = $('tlStops');
  wrap.innerHTML = '';
  stops().forEach(st => {
    const b = document.createElement('button');
    b.dataset.offset = st.offset;
    b.style.left = `${((st.offset - r0) / span) * 100}%`;
    b.innerHTML = `${st.label}<small>${st.sub || '&nbsp;'}</small>`;
    b.addEventListener('click', () => { stopPlay(); setOffset(st.offset); });
    wrap.appendChild(b);
  });

  // คลิก/ลากบนกราฟ = เลือกชั่วโมงใดก็ได้
  const graph = $('tlGraph');
  const pick = e => {
    const r = graph.getBoundingClientRect();
    setOffset(r0 + ((e.clientX - r.left) / r.width) * span);
  };
  graph.onpointerdown = e => { stopPlay(); graph.setPointerCapture(e.pointerId); pick(e); graph.onpointermove = pick; };
  graph.onpointerup = e => { graph.onpointermove = null; graph.releasePointerCapture(e.pointerId); };
  renderTimelineCursor(state.target);
}

function renderTimelineCursor(off) {
  const s = S();
  if (!s) return;
  const [r0, r1] = s.range;
  $('tlCursor').style.left = `${((off - r0) / (r1 - r0)) * 100}%`;
  [...$('tlStops').children].forEach(b => {
    const on = +b.dataset.offset === off;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', on);
  });
  const tr = s.hourly_trend.find(t => t.offset === off);
  if (tr) $('tlLabel').textContent = `${thDateTime(tr.iso)} (${relLabel(off)})`;
}

function togglePlay() { state.playing ? stopPlay() : startPlay(); }
function startPlay() {
  state.playing = true;
  $('playIcon').innerHTML = '<path d="M7 5h4v14H7zM13 5h4v14h-4z"/>';
  $('playBtn').setAttribute('aria-label', 'หยุดเล่น');
  const step = () => {
    const [r0, r1] = S().range;
    let next = state.target + 3;
    if (next > r1) next = r0;
    setOffset(next);
    [3, 6].forEach(k => { const o = next + k; if (o <= r1) loadGrid(o).catch(() => {}); });   // โหลดล่วงหน้า
  };
  step();
  state.playTimer = setInterval(step, 1100);
}
function stopPlay() {
  if (!state.playing) return;
  state.playing = false;
  clearInterval(state.playTimer);
  $('playIcon').innerHTML = '<path d="M8 5v14l11-7z"/>';
  $('playBtn').setAttribute('aria-label', 'เล่นต่อเนื่อง');
}

// --------------------------------------------------------------------------
// พื้นผิวค่าฝุ่น 1 กม.: ขยายภาพ 4 เท่าแบบ bilinear แล้วลงสีตามระดับ
// --------------------------------------------------------------------------
function gridGeom(grid) {
  const { nx, ny, bounds } = grid.grid;
  const dx = (bounds[2] - bounds[0]) / (nx - 1), dy = (bounds[3] - bounds[1]) / (ny - 1);
  // bounds เป็นจุดกึ่งกลางช่อง: ขยายครึ่งช่องให้ภาพครอบพื้นที่ช่องเต็ม
  return { nx, ny, dx, dy, x0: bounds[0] - dx / 2, y0: bounds[1] - dy / 2, x1: bounds[2] + dx / 2, y1: bounds[3] + dy / 2, b: bounds };
}

function buildSurfaceCanvas(grid) {
  const { nx, ny } = grid.grid;
  const vals = grid.pm25;
  const S4 = 4;
  const w = nx * S4, h = ny * S4;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let py = 0; py < h; py++) {
    const gy = Math.max(0, Math.min(ny - 1, (h - 1 - py + 0.5) / S4 - 0.5));   // แถวบนสุด = ทิศเหนือ
    const y0 = Math.min(ny - 2, Math.floor(gy)), fy = gy - y0;
    for (let px = 0; px < w; px++) {
      const gx = Math.max(0, Math.min(nx - 1, (px + 0.5) / S4 - 0.5));
      const x0 = Math.min(nx - 2, Math.floor(gx)), fx = gx - x0;
      const i00 = y0 * nx + x0;
      const v = (vals[i00] * (1 - fx) + vals[i00 + 1] * fx) * (1 - fy) + (vals[i00 + nx] * (1 - fx) + vals[i00 + nx + 1] * fx) * fy;
      const rgb = tierOf(v).rgb;
      const edge = Math.min(px, w - 1 - px, py, h - 1 - py) / (S4 * 4);
      const k = (py * w + px) * 4;
      img.data[k] = rgb[0]; img.data[k + 1] = rgb[1]; img.data[k + 2] = rgb[2];
      img.data[k + 3] = 175 * Math.min(1, edge);
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

const SURF_OPACITY = 0.78;
function initSurfaceLayers() {
  state.surf.beforeId = (map.getStyle().layers || []).find(l => l.type === 'symbol')?.id;
  map.on('sourcedata', e => {
    const q = state.surf.queue?.[e.sourceId];
    if (!q || !q.busy || !e.isSourceLoaded) return;
    q.busy = false;
    if (q.pending) { const p = q.pending; q.pending = null; setSurfaceImage(e.sourceId.slice(3), p.url, p.coords); }
  });
  state.surf.ready = true;
  updateSurface();
}

function setSurfaceImage(k, url, coords) {
  const id = `pm-${k}`;
  const src = map.getSource(id);
  const q = (state.surf.queue = state.surf.queue || {});
  if (src) {
    if (q[id]?.busy) { q[id].pending = { url, coords }; return; }
    q[id] = { busy: true, pending: null };
    src.updateImage({ url, coordinates: coords });
    return;
  }
  q[id] = { busy: true, pending: null };
  map.addSource(id, { type: 'image', url, coordinates: coords });
  map.addLayer({
    id, type: 'raster', source: id,
    paint: { 'raster-opacity': 0, 'raster-opacity-transition': { duration: PREFERS_REDUCED ? 0 : 400, delay: 0 },
             'raster-resampling': 'linear', 'raster-fade-duration': 0 }
  }, state.surf.beforeId);
}
function setSurfaceOpacity(k, v) { if (map.getLayer(`pm-${k}`)) map.setPaintProperty(`pm-${k}`, 'raster-opacity', v); }

function updateSurface() {
  if (!state.surf.ready || !state.grids.has(state.offset)) return;
  const off = state.offset;
  const { active } = state.surf;
  const other = active === 'A' ? 'B' : 'A';
  if (!state.layers.pmSurface) {
    setSurfaceOpacity('A', 0); setSurfaceOpacity('B', 0);
    state.surf.shownKey = null;
    return;
  }
  const url = state.surfaces.get(off);
  if (!url) return;
  if (state.surf.shownKey === off) { setSurfaceOpacity(active, SURF_OPACITY); return; }
  const g = gridGeom(state.grids.get(off));
  const coords = [[g.x0, g.y1], [g.x1, g.y1], [g.x1, g.y0], [g.x0, g.y0]];
  setSurfaceImage(other, url, coords);
  setSurfaceOpacity(other, SURF_OPACITY);
  setSurfaceOpacity(active, 0);
  state.surf.active = other;
  state.surf.shownKey = off;
}

function cellAt(grid, lon, lat) {
  const g = gridGeom(grid);
  const ix = Math.round((lon - g.b[0]) / g.dx), iy = Math.round((lat - g.b[1]) / g.dy);
  if (ix < 0 || iy < 0 || ix >= g.nx || iy >= g.ny) return -1;
  return iy * g.nx + ix;
}

function onMapHoverSurface(e) {
  if (!state.layers.pmSurface || !state.grids.has(state.offset)) return;
  if (overlay.pickObject({ x: e.point.x, y: e.point.y, radius: 4 })) return;
  const grid = current().grid;
  const i = cellAt(grid, e.lngLat.lng, e.lngLat.lat);
  if (i < 0) return hideTip();
  const v = grid.pm25[i], t = tierOf(v);
  showTip({ x: e.point.x, y: e.point.y }, `<div class="tt-sub">ช่องกริด 1×1 กม. (ค่าประมาณ)</div>
    <div class="tt-row"><span class="tt-val" style="color:${t.ink}">${fmt(v)}</span><span>µg/m³ · ${t.name}</span></div>
    <div class="tt-sub">เพดานอากาศ ${grid.mixing_height[i].toLocaleString()} m</div>`);
}

// --------------------------------------------------------------------------
// โดมความสูงชั้นผสมอากาศ
// --------------------------------------------------------------------------
const DOME_VE = 30;      // ขยายแนวตั้ง 30 เท่า เพื่อให้มองเห็นรูปทรง
const DOME_STEP = 3;     // ตาข่ายโดม 3×3 กม. (ข้อมูลอุตุฯ ต้นทางหยาบกว่านี้มาก ละเอียดกว่านี้ไม่ได้ข้อมูลเพิ่ม)

// สีโดมตามความสูงชั้นผสมอากาศ: เพดานต่ำ = ม่วงเข้มทึบ (ฝุ่นถูกกัก) → เพดานสูง = ฟ้าอ่อนโปร่ง (อากาศเปิด)
// ใช้โทนม่วง–ฟ้า ไม่ซ้ำกับสีระดับฝุ่น (ฟ้า/เขียว/เหลือง/ส้ม/แดง) และความทึบลดลงตามความสูง
const DOME_STOPS = [
  { h: 100,  c: [45, 0, 75],     a: 190 },
  { h: 250,  c: [88, 36, 145],   a: 175 },
  { h: 400,  c: [130, 108, 182], a: 155 },
  { h: 600,  c: [72, 148, 204],  a: 130 },
  { h: 850,  c: [128, 204, 236], a: 110 },
  { h: 1200, c: [226, 245, 255], a: 90 },
];
function mixColor(mh) {
  const S = DOME_STOPS;
  if (!(mh > S[0].h)) return [...S[0].c, S[0].a];
  for (let i = 1; i < S.length; i++) {
    if (mh <= S[i].h) {
      const t = (mh - S[i - 1].h) / (S[i].h - S[i - 1].h);
      const lerp = (a, b) => Math.round(a + (b - a) * t);
      return [lerp(S[i - 1].c[0], S[i].c[0]), lerp(S[i - 1].c[1], S[i].c[1]), lerp(S[i - 1].c[2], S[i].c[2]), lerp(S[i - 1].a, S[i].a)];
    }
  }
  const L = S[S.length - 1];
  return [...L.c, L.a];
}

function buildDome(grid) {
  const g = gridGeom(grid);
  const mh = grid.mixing_height, pm = grid.pm25;
  const xs = [], ys = [];
  for (let i = 0; i < g.nx; i += DOME_STEP) xs.push(i);
  if (xs[xs.length - 1] !== g.nx - 1) xs.push(g.nx - 1);
  for (let j = 0; j < g.ny; j += DOME_STEP) ys.push(j);
  if (ys[ys.length - 1] !== g.ny - 1) ys.push(g.ny - 1);
  const W = xs.length, H = ys.length;
  const smooth = t => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
  let bad = 0, lo = Infinity, hi = -Infinity;
  const V = [];
  for (let jj = 0; jj < H; jj++) {
    for (let ii = 0; ii < W; ii++) {
      const k = ys[jj] * g.nx + xs[ii];
      let m = +mh[k];
      if (!Number.isFinite(m)) { m = 0; bad++; }
      lo = Math.min(lo, m); hi = Math.max(hi, m);
      const edge = Math.min(ii, W - 1 - ii, jj, H - 1 - jj) / ((W - 1) * 0.14);
      const taper = smooth(edge);
      V.push([g.b[0] + xs[ii] * g.dx, g.b[1] + ys[jj] * g.dy, m * DOME_VE * taper, m, +pm[k] || 0, taper]);
    }
  }
  const at = (i, j) => V[j * W + i];
  const mPerLon = 111320 * Math.cos(16.85 * Math.PI / 180), mPerLat = 110570;
  const quads = [];
  for (let j = 0; j < H - 1; j++) {
    for (let i = 0; i < W - 1; i++) {
      const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
      const m = (a[3] + b[3] + c[3] + d[3]) / 4;
      const dzdx = ((b[2] + c[2]) - (a[2] + d[2])) / 2 / ((b[0] - a[0]) * mPerLon);
      const dzdy = ((d[2] + c[2]) - (a[2] + b[2])) / 2 / ((d[1] - a[1]) * mPerLat);
      const nz = 1 / Math.hypot(dzdx, dzdy, 1);
      const shade = 0.72 + 0.28 * Math.max(0, (dzdx * 0.6 - dzdy * 0.6) * nz + nz * 0.55);
      const base = mixColor(m);
      const skirt = 0.3 + 0.7 * (a[5] + b[5] + c[5] + d[5]) / 4;     // ขอบที่ลาดลงพื้นให้จางลง
      quads.push({
        poly: [a.slice(0, 3), b.slice(0, 3), c.slice(0, 3), d.slice(0, 3)],
        color: [Math.min(255, base[0] * shade), Math.min(255, base[1] * shade), Math.min(255, base[2] * shade), Math.round(base[3] * skirt)],
        mh: m, pm: (a[4] + b[4] + c[4] + d[4]) / 4
      });
    }
  }
  const lines = [];
  for (let j = 0; j < H; j += 2) lines.push(Array.from({ length: W }, (_, i) => at(i, j).slice(0, 3)));
  for (let i = 0; i < W; i += 2) lines.push(Array.from({ length: H }, (_, j) => at(i, j).slice(0, 3)));
  const info = { quads: quads.length, lo: Math.round(lo), hi: Math.round(hi), bad };
  console.info('[dome]', info);
  return { quads, lines, info };
}

function updateDomeStatus(dome, err) {
  const el = $('domeStatus');
  el.hidden = !state.layers.volume;
  el.classList.toggle('err', !!err);
  if (err) { el.textContent = 'แสดงโดมไม่ได้: ' + err; return; }
  if (!dome) return;
  const { lo, hi, quads, bad } = dome.info;
  el.textContent = quads
    ? `เพดานอากาศ ${lo.toLocaleString()}–${hi.toLocaleString()} m · ตาข่าย ${DOME_STEP} กม. · ขยายแนวตั้ง ${DOME_VE}×` + (bad ? ` · ข้อมูลหาย ${bad} จุด` : '')
    : 'ไม่มีข้อมูลความสูงชั้นผสมอากาศ';
  if (quads && map.getPitch() < 20) el.textContent += ' · กด "3 มิติ" เพื่อเห็นรูปทรงโดม';
}

// --------------------------------------------------------------------------
// ชั้นข้อมูล deck.gl
// --------------------------------------------------------------------------
function updateLayers() {
  if (!state.grids.has(state.offset)) return;
  const { off, grid, snap } = current();
  const L = state.layers;
  const layers = [];
  updateSurface();

  if (L.volume) {
    try {
      const dome = state.domes.get(off) || buildDome(grid);
      state.domes.set(off, dome);
      layers.push(new deck.SolidPolygonLayer({
        id: 'mixing-dome', data: dome.quads, _full3d: true, extruded: false,
        getPolygon: d => d.poly, getFillColor: d => d.color,
        pickable: true, parameters: { depthMask: false },
        onHover: info => info.object ? showTip(info, domeTip(info.object)) : hideTip()
      }));
      layers.push(new deck.PathLayer({
        id: 'mixing-dome-mesh', data: dome.lines, getPath: d => d,
        getColor: [255, 255, 255, 150], widthUnits: 'pixels', getWidth: 1, parameters: { depthMask: false }
      }));
      updateDomeStatus(dome);
    } catch (e) {
      console.error('[dome] build failed', e);
      updateDomeStatus(null, e.message);
    }
  } else {
    $('domeStatus').hidden = true;
  }

  const hs = hotspotsAt();
  if (L.hotspots && hs && hs.points.length) {
    layers.push(new deck.HeatmapLayer({
      id: 'hotspot-heat', data: hs.points.filter(d => d.age <= 48),
      getPosition: d => [d.lon, d.lat],
      getWeight: d => Math.max(1, d.frp_mw) * (d.age <= 24 ? 1 : 0.5),
      radiusPixels: 38, intensity: 1.1, threshold: 0.06,
      updateTriggers: { getWeight: off },
      colorRange: [[255, 237, 160, 0], [254, 178, 76, 90], [253, 141, 60, 140], [240, 59, 32, 180], [189, 0, 38, 210]]
    }));
    layers.push(new deck.ScatterplotLayer({
      id: 'hotspot-points', data: hs.points,
      getPosition: d => [d.lon, d.lat], radiusUnits: 'pixels',
      getRadius: d => 2.5 + Math.min(5, Math.sqrt(Math.max(0, d.frp_mw)) * 0.8),
      getFillColor: d => [214, 40, 40, d.age <= 24 ? 255 : d.age <= 48 ? 170 : 80],
      getLineColor: d => [255, 255, 255, d.age <= 48 ? 255 : 120],
      updateTriggers: { getFillColor: off, getLineColor: off },
      stroked: true, lineWidthUnits: 'pixels', getLineWidth: 1.2,
      pickable: true, parameters: { depthTest: false },
      onHover: info => info.object ? showTip(info, hotspotTip(info.object)) : hideTip()
    }));
  }

  if (L.stations) {
    layers.push(new deck.ScatterplotLayer({
      id: 'stations', data: snap.stations,
      getPosition: d => [d.lon, d.lat], radiusUnits: 'pixels',
      getRadius: d => (d.point_id === state.selected ? 15 : 11),
      getFillColor: d => [...tierOf(d.pm25).rgb, 255],
      // ขอบสีส้ม = สถานีที่ไม่มีค่าตรวจวัดล่าสุด (ค่าเริ่มต้นจากค่าเฉลี่ยจังหวัด)
      // ขอบเขียวเข้ม = มีค่าเรียลไทม์ · ขอบส้ม = ไม่มีค่าตรวจวัดล่าสุด · ขอบขาว = ค่าจากโมเดล
      getLineColor: d => (d.point_id === state.selected ? [27, 40, 56, 255] : d.source === 'realtime' ? [10, 110, 80, 255]
        : d.lag_source !== 'obs' ? [192, 100, 0, 230] : [255, 255, 255, 255]),
      stroked: true, lineWidthUnits: 'pixels', getLineWidth: d => (d.point_id === state.selected ? 3 : d.source === 'realtime' ? 3 : 2),
      pickable: true, autoHighlight: true, highlightColor: [27, 40, 56, 60],
      parameters: { depthTest: false },
      transitions: { getFillColor: { duration: 400 } },
      updateTriggers: { getFillColor: off, getRadius: state.selected, getLineColor: [state.selected, off], getLineWidth: [state.selected, off] },
      onClick: info => info.object && openDrawer(info.object.point_id),
      onHover: info => info.object ? showTip(info, stationTip(info.object)) : hideTip()
    }));
    layers.push(new deck.TextLayer({
      id: 'station-values', data: snap.stations,
      getPosition: d => [d.lon, d.lat], getText: d => String(Math.round(d.pm25)),
      getSize: 11, getColor: [27, 40, 56, 255],
      fontFamily: 'IBM Plex Sans Thai, sans-serif', fontWeight: 600, characterSet: '0123456789',
      parameters: { depthTest: false }, getTextAnchor: 'middle', getAlignmentBaseline: 'center',
      updateTriggers: { getText: off }
    }));
  }
  overlay.setProps({ layers });
}

// จุดความร้อน ณ เวลาที่เลือก: แสดงย้อนหลัง 5 วันจากเวลานั้น (ไม่แสดงจุดที่เกิดภายหลัง)
function hotspotsAt() {
  const hs = state.hotspots;
  if (!hs || hs.status !== 'ok' || !state.grids.has(state.offset)) return null;
  const ref = parseLocal(current().snap.target_time).getTime() + 3600e3;   // นับรวมทั้งชั่วโมงที่เลือก
  const points = [];
  const daily = [0, 0, 0, 0, 0];
  for (const p of hs.hotspots) {
    const age = (ref - p.t * 1000) / 3600e3;
    if (age < 0 || age > 120) continue;
    points.push({ ...p, age });
    if (p.in_province) daily[Math.min(4, Math.floor(age / 24))]++;
  }
  return { points, daily };
}

function domeTip(o) {
  const m = Math.round(o.mh);
  const note = m < 300 ? 'เพดานต่ำ ฝุ่นถูกกักใกล้พื้น' : m < 600 ? 'เพดานปานกลาง' : 'เพดานสูง ฝุ่นกระจายตัวได้ดี';
  return `<div class="tt-title">ความสูงชั้นผสมอากาศ (เพดานอากาศ)</div>
    <div class="tt-val">${m.toLocaleString()} m</div>
    <div class="tt-sub">${note} · ค่าฝุ่นใต้โดม ${fmt(o.pm)} µg/m³</div>
    <div class="tt-sub">ความสูงโดมขยาย ${DOME_VE} เท่าเพื่อให้เห็นรูปทรง</div>`;
}

function hotspotTip(d) {
  const ago = d.age < 1 ? 'ไม่ถึง 1 ชม.ก่อน' : d.age < 24 ? `${Math.round(d.age)} ชม.ก่อน` : `${Math.floor(d.age / 24)} วันก่อน`;
  const sat = { N: 'Suomi NPP', '1': 'NOAA-20', N20: 'NOAA-20', '2': 'NOAA-21', N21: 'NOAA-21', T: 'Terra', A: 'Aqua' }[d.satellite] || d.satellite;
  return `<div class="tt-title">จุดความร้อน · ${ago}เวลาที่เลือก</div>
    <div class="tt-sub">${thDateTime(d.time_local)} · ${sat} (${d.daynight === 'N' ? 'กลางคืน' : 'กลางวัน'})</div>
    <div class="tt-row">กำลังแผ่รังสี <b>${d.frp_mw} MW</b> · ความเชื่อมั่น${d.confidence}</div>
    <div class="tt-sub">${d.in_province ? d.district : 'นอกเขตจังหวัด (กันชน 50 กม.)'}</div>`;
}

function stationTip(s) {
  const t = tierOf(s.pm25);
  return `<div class="tt-title">${shortName(s.name)}</div>
    <div class="tt-sub">${s.district}</div>
    <div class="tt-row"><span class="tt-val" style="color:${t.ink}">${fmt(s.pm25)}</span><span class="tier-badge" style="--tier:${t.css}">${t.name}</span></div>
    <div class="tt-sub">${s.source === 'realtime' ? `ค่าตรวจวัดเรียลไทม์ ${thTime(parseLocal(s.realtime.time))} · ${s.realtime.source}`
      : s.source === 'obs' ? 'ค่าตรวจวัดจริง (เฉลี่ยรายวัน)' : 'ค่าเฉลี่ยรายวันจากโมเดล'}</div>
    ${s.source === 'realtime' ? `<div class="tt-sub">โมเดล (เฉลี่ยทั้งวัน): ${fmt(s.model_daily)} µg/m³</div>` : ''}
    ${s.lag_source !== 'obs' ? `<div class="tt-sub" style="color:#8a5a00">ไม่มีค่าตรวจวัดล่าสุด${s.last_obs ? ' (ครั้งสุดท้าย ' + thDate(new Date(s.last_obs + 'T00:00')) + ')' : ''} — เริ่มจากค่าเฉลี่ยจังหวัด</div>` : ''}
    <div class="tt-sub">คลิกเพื่อดูกราฟ</div>`;
}

function showTip(info, html) {
  const tip = $('mapTip');
  if (!html) return hideTip();
  tip.innerHTML = html;
  tip.style.display = 'block';
  const rect = map.getCanvas().getBoundingClientRect();
  let x = rect.left + info.x + 14, y = rect.top + info.y + 14;
  if (x + tip.offsetWidth > window.innerWidth - 8) x -= tip.offsetWidth + 28;
  if (y + tip.offsetHeight > window.innerHeight - 8) y -= tip.offsetHeight + 28;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
function hideTip() { $('mapTip').style.display = 'none'; }

// --------------------------------------------------------------------------
// ลิ้นชักสถานี: กราฟรายชั่วโมงของสถานีจริง (ทั้งพยากรณ์และย้อนหลัง)
// --------------------------------------------------------------------------
function openDrawer(pid) {
  state.selected = pid;
  $('stationDrawer').classList.add('open');
  $('stationDrawer').setAttribute('aria-hidden', 'false');
  renderDrawer();
  updateLayers();
}
function closeDrawer() {
  state.selected = null;
  $('stationDrawer').classList.remove('open');
  $('stationDrawer').setAttribute('aria-hidden', 'true');
  updateLayers();
}

function renderDrawer() {
  const { off, snap } = current();
  const s = snap.stations.find(x => x.point_id === state.selected);
  if (!s) return;
  const t = tierOf(s.pm25);
  $('stDistrict').textContent = s.district;
  $('stName').textContent = shortName(s.name);
  $('stName').title = s.name;
  $('stHero').style.setProperty('--tier', t.css);
  $('stHero').style.setProperty('--tier-ink', t.ink);
  tweenNumber($('stValue'), s.pm25, 350);
  $('stBadge').textContent = t.name;
  $('stWhen').textContent = off === 0 ? 'ตอนนี้' : thDateTime(snap.target_time);

  const ser = S().station_hourly[s.point_id] || [];
  const [r0] = S().range;
  const valAt = o => ser[o - r0];
  const hz = $('stHorizons');
  const st = stops();
  hz.style.gridTemplateColumns = `repeat(${st.length}, 1fr)`;
  hz.innerHTML = '';
  st.forEach(p => {
    const v = valAt(p.offset);
    const b = document.createElement('button');
    b.style.setProperty('--c', v != null ? tierOf(v).hex : '#ddd');
    b.className = p.offset === off ? 'active' : '';
    b.innerHTML = `<span>${p.label}</span><b>${v != null ? Math.round(v) : '–'}</b>`;
    b.addEventListener('click', () => { stopPlay(); setOffset(p.offset); });
    hz.appendChild(b);
  });
  $('chartTitle').textContent = state.mode === 'history' ? 'ย้อนหลัง 7 วัน (ค่าเฉลี่ยรายวัน)' : 'พยากรณ์ 3 วัน (ค่าเฉลี่ยรายวัน)';
  $('chartNote').textContent = (s.source === 'realtime' ? `ตัวเลขด้านบนเป็นค่าเรียลไทม์ ${thTime(parseLocal(s.realtime.time))} (${s.realtime.source}) · กราฟเป็นค่าเฉลี่ยรายวันจากโมเดล`
      : s.source === 'obs' ? 'วันนี้เป็นค่าตรวจวัดจริง' : 'ค่าจากโมเดลรายวัน (เดินหน้าทีละวันจากค่าตรวจวัดล่าสุด)')
    + (s.lag_source !== 'obs' ? ' · สถานีนี้ไม่มีค่าตรวจวัดล่าสุด เริ่มจากค่าเฉลี่ยจังหวัด' : '') + ' · เส้นประ = เวลาที่เลือก';
  const adv = $('stAdvice');
  adv.style.setProperty('--c', t.hex);
  adv.textContent = t.advice;
  renderStationChart(ser);
}

function renderStationChart(ser) {
  if (!ser.length || typeof Chart === 'undefined') return;
  const [r0, r1] = S().range;
  const data = ser.map((v, i) => ({ x: r0 + i, y: v }));
  const yMax = Math.max(40, Math.ceil(Math.max(...ser) / 10) * 10 + 5);
  const cursor = state.offset;
  const trend = S().hourly_trend;
  const bands = {
    id: 'tierBands',
    beforeDraw(chart) {
      const { ctx, chartArea: a, scales: { x, y } } = chart;
      if (!a) return;
      ctx.save();
      let lo = 0;
      TIERS.forEach(t => {
        const hi = Math.min(t.max, yMax);
        if (lo < yMax) { ctx.fillStyle = t.hex + '2a'; ctx.fillRect(a.left, y.getPixelForValue(hi), a.right - a.left, y.getPixelForValue(lo) - y.getPixelForValue(hi)); }
        lo = t.max;
      });
      const cx = x.getPixelForValue(cursor);
      ctx.strokeStyle = '#1b2838'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(cx, a.top); ctx.lineTo(cx, a.bottom); ctx.stroke();
      ctx.restore();
    }
  };
  const hist = state.mode === 'history';
  const cfg = {
    type: 'line',
    data: { datasets: [{ data, borderColor: '#4a5a6c', borderWidth: 1.8, pointRadius: 0, tension: 0, stepped: true, fill: false,
      segment: { borderColor: c => darken(tierOf((c.p0.parsed.y + c.p1.parsed.y) / 2).rgb) } }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: PREFERS_REDUCED ? false : { duration: 300 },
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      onClick: (e, els, chart) => { const o = Math.round(chart.scales.x.getValueForPixel(e.x)); stopPlay(); setOffset(o); },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#fff', titleColor: '#1b2838', bodyColor: '#1b2838', borderColor: '#dfe5ec', borderWidth: 1,
          callbacks: {
            title: it => { const tr = trend.find(t => t.offset === it[0].parsed.x); return tr ? thDateTime(tr.iso) : ''; },
            label: it => `${it.parsed.y.toFixed(1)} µg/m³`
          }
        }
      },
      scales: {
        x: { type: 'linear', min: r0, max: r1, grid: { display: false },
             ticks: { stepSize: hist ? 24 : 12, color: '#7a8898', font: { size: 10 },
                      callback: v => (v === 0 ? 'ตอนนี้' : hist ? `${-v / 24} วัน` : `+${v}ชม.`) } },
        y: { min: 0, max: yMax, grid: { color: 'rgba(27,40,56,.06)' }, ticks: { color: '#7a8898', font: { size: 10 } } }
      }
    },
    plugins: [bands]
  };
  if (stationChart) stationChart.destroy();
  stationChart = new Chart($('stationChart'), cfg);
}

// --------------------------------------------------------------------------
// เส้นลมแบบ Windy (สนามลมจากกริด 1 กม.)
// --------------------------------------------------------------------------
const WIND_BINS = [1, 2, 3, 4.5, 6, Infinity];
const WIND_COLORS = ['#8cc4e3', '#4596cc', '#1f62ad', '#4434a6', '#7a249d', '#b0127a'];
const WIND_LABELS = ['< 1', '1–2', '2–3', '3–4.5', '4.5–6', '> 6'];
const windBin = s => { const i = WIND_BINS.findIndex(b => s < b); return i < 0 ? WIND_BINS.length - 1 : i; };

const Wind = (() => {
  let canvas, ctx, dpr = 1, particles = [], field = null, geom = null, raf = null, enabled = true;
  const N = PREFERS_REDUCED ? 0 : (window.innerWidth <= 820 ? 650 : 1300);
  const TRAIL = 34;
  const LEVEL_ALPHA = [0.16, 0.42, 0.85];

  function init() {
    canvas = $('wind-canvas');
    ctx = canvas.getContext('2d');
    resize();
    window.addEventListener('resize', resize);
    map.on('move', () => { particles.forEach(p => { p.tr.length = 0; }); if (PREFERS_REDUCED) drawStatic(); });
  }
  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
  }
  function spawn(p) {
    let w = geom.b[0], s = geom.b[1], e = geom.b[2], n = geom.b[3];
    try {
      const vb = map.getBounds();
      const w2 = Math.max(w, vb.getWest()), e2 = Math.min(e, vb.getEast()), s2 = Math.max(s, vb.getSouth()), n2 = Math.min(n, vb.getNorth());
      if (w2 < e2 && s2 < n2) { w = w2; e = e2; s = s2; n = n2; }
    } catch (_) { /* ignore */ }
    p.lon = w + Math.random() * (e - w); p.lat = s + Math.random() * (n - s);
    p.age = 0; p.max = 90 + Math.random() * 170; p.dying = false; p.bin = 0;
    p.tr = p.tr || []; p.tr.length = 0;
    return p;
  }
  function sample(lon, lat) {
    const gx = (lon - geom.b[0]) / geom.dx, gy = (lat - geom.b[1]) / geom.dy;
    if (gx < 0 || gy < 0 || gx > geom.nx - 1 || gy > geom.ny - 1) return null;
    const x0 = Math.min(geom.nx - 2, Math.floor(gx)), y0 = Math.min(geom.ny - 2, Math.floor(gy));
    const fx = gx - x0, fy = gy - y0, i = y0 * geom.nx + x0, nx = geom.nx;
    const bl = a => (a[i] * (1 - fx) + a[i + 1] * fx) * (1 - fy) + (a[i + nx] * (1 - fx) + a[i + nx + 1] * fx) * fy;
    return [bl(field.u), bl(field.v)];
  }
  function setField(grid) {
    if (!grid || !grid.u) return;
    field = grid; geom = gridGeom(grid);
    if (!particles.length) particles = Array.from({ length: N }, () => { const p = spawn({}); p.age = Math.random() * p.max * 0.6; return p; });
    if (PREFERS_REDUCED) { drawStatic(); return; }
    if (!raf && enabled) raf = requestAnimationFrame(frame);
  }
  function frame() {
    raf = null;
    if (!enabled || !field) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const k = 0.0016 * Math.pow(2, 9 - map.getZoom());     // ความเร็วบนจอคงที่ทุกระดับซูม (เชิงภาพ)
    const paths = WIND_COLORS.map(() => [new Path2D(), new Path2D(), new Path2D()]);
    for (const p of particles) {
      if (!p.dying) {
        const uv = sample(p.lon, p.lat);
        if (!uv || ++p.age > p.max) p.dying = true;
        else {
          p.lon += (uv[0] * k) / Math.cos(p.lat * Math.PI / 180);
          p.lat += uv[1] * k;
          p.bin = windBin(Math.hypot(uv[0], uv[1]));
          const q = map.project([p.lon, p.lat]);
          if (q.x < -30 || q.y < -30 || q.x > innerWidth + 30 || q.y > innerHeight + 30) p.dying = true;
          else { p.tr.push(q.x * dpr, q.y * dpr); if (p.tr.length > TRAIL * 2) p.tr.splice(0, 2); }
        }
      }
      if (p.dying) { p.tr.splice(0, 4); if (p.tr.length < 4) { spawn(p); continue; } }
      const n = p.tr.length / 2;
      if (n < 2) continue;
      for (let lvl = 0; lvl < 3; lvl++) {
        const a = Math.floor(((n - 1) * lvl) / 3), b = Math.floor(((n - 1) * (lvl + 1)) / 3);
        if (b <= a) continue;
        const path = paths[p.bin][lvl];
        path.moveTo(p.tr[a * 2], p.tr[a * 2 + 1]);
        for (let i = a + 1; i <= b; i++) path.lineTo(p.tr[i * 2], p.tr[i * 2 + 1]);
      }
    }
    ctx.lineWidth = 1.6 * dpr; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (let c = 0; c < WIND_COLORS.length; c++) {
      ctx.strokeStyle = WIND_COLORS[c];
      for (let lvl = 0; lvl < 3; lvl++) { ctx.globalAlpha = LEVEL_ALPHA[lvl]; ctx.stroke(paths[c][lvl]); }
    }
    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(frame);
  }
  function drawStatic() {
    if (!field || !enabled) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 1.5 * dpr;
    for (let j = 0; j < geom.ny; j += 6) {
      for (let i = 0; i < geom.nx; i += 6) {
        const k = j * geom.nx + i, u = field.u[k], v = field.v[k], spd = Math.hypot(u, v);
        const p = map.project([geom.b[0] + i * geom.dx, geom.b[1] + j * geom.dy]);
        const len = 12 + spd * 4, ang = Math.atan2(-v, u);
        const x2 = p.x + Math.cos(ang) * len, y2 = p.y + Math.sin(ang) * len;
        ctx.strokeStyle = WIND_COLORS[windBin(spd)];
        ctx.beginPath();
        ctx.moveTo(p.x * dpr, p.y * dpr); ctx.lineTo(x2 * dpr, y2 * dpr);
        ctx.moveTo((x2 - Math.cos(ang - 0.5) * 5) * dpr, (y2 - Math.sin(ang - 0.5) * 5) * dpr);
        ctx.lineTo(x2 * dpr, y2 * dpr);
        ctx.lineTo((x2 - Math.cos(ang + 0.5) * 5) * dpr, (y2 - Math.sin(ang + 0.5) * 5) * dpr);
        ctx.stroke();
      }
    }
  }
  function setEnabled(on) {
    enabled = on;
    if (!on) { cancelAnimationFrame(raf); raf = null; ctx.clearRect(0, 0, canvas.width, canvas.height); }
    else setField(field);
  }
  return { init, setField, setEnabled };
})();

// --------------------------------------------------------------------------
// ปุ่มและคีย์ลัด
// --------------------------------------------------------------------------
function bindUI() {
  $('playBtn').addEventListener('click', togglePlay);
  $('closeDrawerBtn').addEventListener('click', closeDrawer);
  $('modeForecast').addEventListener('click', () => setMode('forecast'));
  $('modeHistory').addEventListener('click', () => setMode('history'));

  document.querySelectorAll('[data-layer]').forEach(cb => {
    cb.addEventListener('change', e => {
      const key = e.target.dataset.layer;
      state.layers[key] = e.target.checked;
      if (key === 'wind') { Wind.setEnabled(e.target.checked); $('windLegend').hidden = !e.target.checked; }
      if (key === 'volume') {
        $('domeLegend').hidden = !e.target.checked;
        if (e.target.checked && map.getPitch() < 20) setView('3D');
      }
      updateLayers();
    });
  });
  $('view2D').addEventListener('click', () => setView('2D'));
  $('view3D').addEventListener('click', () => setView('3D'));
  map.on('pitchend', () => { if (state.layers.volume) updateDomeStatus(state.domes.get(state.offset)); });

  const dlg = $('accuracyDialog');
  $('openAccuracyBtn').addEventListener('click', () => { renderAccuracy(); dlg.showModal(); });
  $('closeAccuracyBtn').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });

  if (window.innerWidth <= 820) {
    $('summaryPanel').classList.add('collapsed');
    $('sheetHandle').setAttribute('aria-expanded', 'false');
  }
  $('layersToggle').addEventListener('click', () => {
    if (window.innerWidth > 820) return;
    $('layersToggle').setAttribute('aria-expanded', $('layerPanel').classList.toggle('expanded'));
  });
  $('sheetHandle').addEventListener('click', () => {
    const collapsed = $('summaryPanel').classList.toggle('collapsed');
    $('sheetHandle').setAttribute('aria-expanded', !collapsed);
  });

  // ← → ทีละชั่วโมง, Shift+← → ทีละวัน, เว้นวรรค เล่น/หยุด, Esc ปิด
  document.addEventListener('keydown', e => {
    if (e.target.closest('input, textarea, dialog[open]') || !S()) return;
    const step = e.shiftKey ? 24 : 1;
    if (e.key === 'ArrowRight') { e.preventDefault(); stopPlay(); setOffset(state.target + step); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); stopPlay(); setOffset(state.target - step); }
    else if (e.key === ' ' && !e.target.closest('button')) { e.preventDefault(); togglePlay(); }
    else if (e.key === 'Escape' && state.selected) closeDrawer();
  });
}

function setView(mode) {
  const is3D = mode === '3D';
  $('view3D').classList.toggle('active', is3D); $('view3D').setAttribute('aria-pressed', is3D);
  $('view2D').classList.toggle('active', !is3D); $('view2D').setAttribute('aria-pressed', !is3D);
  map.easeTo({ pitch: is3D ? 60 : 0, bearing: is3D ? -18 : 0, duration: PREFERS_REDUCED ? 0 : 1000 });
  // 3 มิติ = เปิดโดม, 2 มิติ = ปิดโดม (มองจากด้านบนโดมจะบังพื้นผิวค่าฝุ่น)
  if (is3D !== state.layers.volume) {
    state.layers.volume = is3D;
    document.querySelector('[data-layer="volume"]').checked = is3D;
    $('domeLegend').hidden = !is3D;
    updateLayers();
  }
}

function renderAccuracy() {
  const bm = state.series.forecast?.backtest_benchmark;
  if (!bm) return;
  if (bm.test_samples) $('accSamples').textContent = bm.test_samples.toLocaleString();
  $('accTierTable').querySelector('tbody').innerHTML = Object.values(bm.tier_accuracy).map((t, i) => `<tr>
      <td><span class="tier-cell"><i style="background:${TIERS[i].hex}"></i>${TIERS[i].name}</span></td>
      <td>${TIERS[i].range}</td>
      <td><span class="acc-bar"><i style="width:${t.f1_pct * 0.6}px"></i><b>${t.f1_pct}%</b></span></td>
      <td>${t.recall_pct}%</td><td>${t.precision_pct}%</td></tr>`).join('');
  const hc = bm.horizon_confidence || {};
  $('accHorizons').innerHTML = FC_STOPS.map(h => {
    const c = hc[`plus_${h}h`];
    if (!c) return '<div class="hz"></div>';
    return `<div class="hz" title="RMSE ±${c.rmse} µg/m³"><b>${c.accuracy_pct}%</b>
      <i style="height:${(c.accuracy_pct - 50) * 1.6}px; opacity:${state.offset === h ? 1 : .55}"></i>
      <span>${h === 0 ? 'ตอนนี้' : '+' + h + ' ชม.'}<br>±${c.rmse}</span></div>`;
  }).join('');
}

function buildLegend() {
  $('legendScale').innerHTML = TIERS.map(t => `<div><i style="background:${t.hex}"></i><b>${t.range}</b>${t.name}</div>`).join('');
  $('windLegend').innerHTML = WIND_COLORS.map((c, i) => `<div><i style="background:${c}"></i>${WIND_LABELS[i]}</div>`).join('') + '<span>m/s</span>';
  $('domeLegend').innerHTML = [150, 300, 450, 600, 800, 1000].map(m => {
    const c = mixColor(m);
    return `<div><i style="background:rgba(${c[0]},${c[1]},${c[2]},${(c[3] / 255).toFixed(2)})"></i>${m === 1000 ? '1000+' : m}</div>`;
  }).join('') + '<span>m</span>';
}

// --------------------------------------------------------------------------
// ตัวช่วย
// --------------------------------------------------------------------------
function fmt(v) { return (Math.round(v * 10) / 10).toFixed(1); }
function tweenNumber(el, to, dur) {
  const from = parseFloat(el.dataset.v ?? to);
  el.dataset.v = to;
  if (PREFERS_REDUCED || !isFinite(from) || from === to) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now - t0) / dur);
    el.textContent = fmt(from + (to - from) * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
function darken(rgb) { return `rgb(${rgb.map(c => Math.round(c * 0.72)).join(',')})`; }
function compassTH(deg) {
  const dirs = ['เหนือ', 'ตะวันออกเฉียงเหนือ', 'ตะวันออก', 'ตะวันออกเฉียงใต้', 'ใต้', 'ตะวันตกเฉียงใต้', 'ตะวันตก', 'ตะวันตกเฉียงเหนือ'];
  return dirs[Math.round((((deg % 360) + 360) % 360) / 45) % 8];
}
function shortName(name) { return name.replace(/\s*จ\.?\s*พิษณุโลก\s*$/, '').trim(); }
function parseLocal(s) { return new Date(s.replace(' ', 'T')); }
function thDate(d) { return d.toLocaleDateString('th-TH', { day: 'numeric', month: 'short', year: 'numeric' }); }
function thTime(d) { return d.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' }) + ' น.'; }
function thDateTime(s) {
  const d = parseLocal(s);
  return `${d.toLocaleDateString('th-TH', { weekday: 'short', day: 'numeric', month: 'short' })} ${thTime(d)}`;
}
function thShortDT(s) {
  const d = parseLocal(s);
  return `${d.toLocaleDateString('th-TH', { weekday: 'short' })} ${String(d.getHours()).padStart(2, '0')}:00`;
}
let toastTimer;
function showToast(msg, ms = 5000) {
  const t = $('toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer);
  if (ms) toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

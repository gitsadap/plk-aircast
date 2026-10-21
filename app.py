"""
PLK AirCast — backend v9 (รองรับ Hugging Face Spaces)

หลักการ: ให้ข้อมูลที่ป้อนโมเดลตอนใช้งานจริง "ตรงกับ" ตอนฝึกให้มากที่สุด
  - โมเดลเป็นโมเดลรายวัน -> คำนวณค่าอากาศเฉลี่ยรายวัน (ตามวันเวลาไทย) ก่อนป้อน
  - จุดความร้อนใช้ MODIS (ชุดเดียวกับ ee.ImageCollection("FIRMS") ที่ใช้ตอนฝึก) นับตามวันปฏิทิน
  - LST ตอนฝึกหายไป 74% -> ตั้ง lst_missing=1 และใช้ค่าล่าสุดของสถานี (เหมือน ffill ตอนฝึก)
  - prov_pm25_mean_lag_1 = ค่าเฉลี่ยทั้งจังหวัดของวันก่อนหน้า (ตามนิยามตอนฝึก)
  - ค่า lag ของ PM2.5 เดินหน้าแบบ recursive ทีละวันจากวันสุดท้ายที่มีค่าตรวจวัด
"""
import os
import time
# เซิร์ฟเวอร์คลาวด์มักตั้งเวลาเป็น UTC แต่โค้ดนี้ใช้นาฬิกาเครื่องเป็นเวลาไทย -> บังคับเป็น Asia/Bangkok
os.environ["TZ"] = os.environ.get("APP_TZ", "Asia/Bangkok")
if hasattr(time, "tzset"):
    time.tzset()
import csv
import io
import json
import math
import joblib
import requests
import numpy as np
import warnings
import traceback
import re
import threading
from datetime import datetime, date, timedelta, timezone
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# โมเดลฝึกด้วย DataFrame แต่ตอนใช้งานป้อน array ตามลำดับคอลัมน์เดียวกัน — คำเตือนนี้ไม่มีผลต่อผลลัพธ์
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI(title="Phitsanulok PM2.5 Spatio-Temporal Forecast API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def no_stale_static(request, call_next):
    """บังคับให้เบราว์เซอร์ตรวจไฟล์ใหม่ทุกครั้ง (กันปัญหาใช้ app.js เวอร์ชันเก่าค้างใน cache)"""
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.exception_handler(Exception)
async def explain_errors(request, exc):
    """ส่งรายละเอียดข้อผิดพลาดกลับไปให้หน้าเว็บแสดง (แทน 500 ที่ไม่มีข้อมูล)"""
    tb = traceback.format_exc()
    print(f"[ERROR] {request.url.path}\n{tb}")
    last = [ln.strip() for ln in tb.strip().splitlines() if ln.strip()][-3:]
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}", "where": request.url.path, "trace": last})


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pm25_model.joblib")
STATIONS_PATH = os.path.join(BASE_DIR, "stations.json")
CSV_PATH = os.path.join(BASE_DIR, "pm25.csv")
FIRMS_KEY_PATH = os.path.join(BASE_DIR, "firms_key.txt")
GISTDA_KEY_PATH = os.path.join(BASE_DIR, "gistda_key.txt")
STATIC_DIR = os.path.join(BASE_DIR, "static")

TZ_TH = timezone(timedelta(hours=7))
HISTORY_HOURS = 168          # สืบค้นย้อนหลัง 7 วัน
FORECAST_HOURS = 72
MAX_ROLLOUT_DAYS = 85        # Open-Meteo ให้ข้อมูลย้อนหลังได้ไม่เกิน 92 วัน
API_VERSION = 8             # หน้าเว็บตรวจเลขนี้ ถ้าไม่ตรงแปลว่าเซิร์ฟเวอร์ที่รันอยู่เป็นตัวเก่า
STALE_DAYS = 7               # สถานีที่ไม่มีค่าตรวจวัดเกินกี่วันก่อนวันล่าสุดของข้อมูล ถือว่า "ข้อมูลเก่า"

# =============================================================================
# ข้อมูลพื้นฐาน
# =============================================================================
model_data = None
stations_data = []
if os.path.exists(MODEL_PATH):
    model_data = joblib.load(MODEL_PATH)
    print("Loaded ML model from", MODEL_PATH)
if os.path.exists(STATIONS_PATH):
    with open(STATIONS_PATH, "r", encoding="utf-8") as f:
        stations_data = json.load(f)
    print(f"Loaded {len(stations_data)} stations")
ST_LAT = np.array([s["lat"] for s in stations_data], float)
ST_LON = np.array([s["lon"] for s in stations_data], float)
ST_IDS = [str(s["point_id"]) for s in stations_data]
N_ST = len(stations_data)


REALTIME_CONFIG_PATH = os.path.join(BASE_DIR, "realtime_sources.json")
REALTIME_LOG_PATH = os.path.join(BASE_DIR, "realtime_log.csv")
REALTIME_POLL_MIN = 20          # ดึงค่าเรียลไทม์ทุก 20 นาที
REALTIME_FRESH_HOURS = 3        # ค่าที่เก่ากว่านี้ไม่ถือว่าเป็น "ตอนนี้"
DAILY_MIN_HOURS = 12            # วันที่จะนับเป็นค่าตรวจวัดรายวัน ต้องมีข้อมูลอย่างน้อยกี่ชั่วโมง


def load_csv_observations():
    """อ่าน pm25.csv (ค่าเฉลี่ยรายวัน) และ LST ล่าสุดของแต่ละสถานี — กรองแบบเดียวกับตอนฝึก (0 < pm25 <= 350)"""
    obs = {pid: {} for pid in ST_IDS}
    lst_last = {pid: (None, None) for pid in ST_IDS}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("point_id", "")).strip()
                try:
                    d = date.fromisoformat(row["date"][:10])
                    v = float(row["pm25"])
                except (KeyError, ValueError):
                    continue
                if 0 < v <= 350 and pid in obs:
                    obs[pid][d] = v
                if pid in lst_last:
                    try:
                        lv = float(row.get("lst_celsius") or 0)
                    except ValueError:
                        lv = 0
                    if lv > 0 and (lst_last[pid][0] is None or d > lst_last[pid][0]):
                        lst_last[pid] = (d, lv)
    lst = {pid: (lv if lv is not None else 26.0) for pid, (_, lv) in lst_last.items()}   # 26.0 = fillna ตอนฝึก
    return obs, lst


CSV_OBS, LST_LAST = load_csv_observations()
OBS, PROV_MEAN_OBS, DATA_AS_OF, LAST_OBS = {}, {}, None, {}
RT_LATEST = {}          # point_id -> {"value", "time", "source"} ค่าล่าสุดจากแหล่งเรียลไทม์
RT_STATUS = {"last_poll": None, "sources": [], "log_rows": 0}
_rt_lock = threading.Lock()


def read_realtime_log():
    """อ่าน realtime_log.csv -> {pid: {datetime_hour: value}} (ค่าเดียวต่อชั่วโมง ใช้ค่าล่าสุด)"""
    out = {}
    if not os.path.exists(REALTIME_LOG_PATH):
        return out
    with open(REALTIME_LOG_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t = datetime.strptime(row["time"], "%Y-%m-%d %H:%M")
                v = float(row["pm25"])
            except (KeyError, ValueError):
                continue
            if 0 <= v <= 1000:
                out.setdefault(row["point_id"], {})[t.replace(minute=0)] = v
    return out


def refresh_observations():
    """รวมค่าตรวจวัดจาก pm25.csv กับค่าเฉลี่ยรายวันที่สะสมจากระบบเรียลไทม์
    วันที่ผ่านไปแล้วและมีข้อมูลอย่างน้อย DAILY_MIN_HOURS ชั่วโมง นับเป็นค่าตรวจวัดรายวัน (ใช้เป็น lag ของโมเดลได้)"""
    global OBS, PROV_MEAN_OBS, DATA_AS_OF, LAST_OBS
    obs = {pid: dict(v) for pid, v in CSV_OBS.items()}
    today = datetime.now(TZ_TH).date()
    log = read_realtime_log()
    for pid, series in log.items():
        if pid not in obs:
            continue
        by_day = {}
        for t, v in series.items():
            by_day.setdefault(t.date(), []).append(v)
        for d, vals in by_day.items():
            if d < today and len(vals) >= DAILY_MIN_HOURS and d not in obs[pid]:
                m = float(np.mean(vals))
                if 0 < m <= 350:
                    obs[pid][d] = m
    prov = {}
    for pid, days in obs.items():
        for d, v in days.items():
            prov.setdefault(d, []).append(v)
    OBS = obs
    PROV_MEAN_OBS = {d: float(np.mean(v)) for d, v in prov.items()}
    DATA_AS_OF = max(PROV_MEAN_OBS) if PROV_MEAN_OBS else None
    LAST_OBS = {pid: (max(obs[pid]) if obs[pid] else None) for pid in ST_IDS}
    RT_STATUS["log_rows"] = sum(len(v) for v in log.values())


refresh_observations()
print("Observation data as of:", DATA_AS_OF,
      "| stations with recent data:", sum(1 for d in LAST_OBS.values() if d and DATA_AS_OF and (DATA_AS_OF - d).days <= STALE_DAYS))


# =============================================================================
# PM2.5 เรียลไทม์รายสถานี
# =============================================================================
DUSTBOY_SOURCE = {
    "name": "DustBoy (CMU CCDC)",
    "type": "json",
    "enabled": True,
    "url": "https://open-api.cmuccdc.org/api/dustboy/station",
    "headers": {"Authorization": "Bearer {api_key}"},
    "api_key_file": "cmuccdc_key.txt",
    "per_station": False,
    "applies_to": "^[0-9]+$",
    "list_path": "auto", "id_path": "auto", "value_path": "auto", "time_path": "auto", "time_format": "auto",
    "match_km": 0,
    "value_url": "",
    "_หมายเหตุ": "API key อยู่ในไฟล์ cmuccdc_key.txt | รหัสสถานีใน stations.json = ไอดี DustBoy จึงจับคู่ด้วยรหัส | path = auto คือให้ระบบหาช่องข้อมูลเอง ตรวจผลที่ /api/realtime/probe | ถ้ารายชื่อสถานีไม่มีค่าฝุ่น ให้ใส่ value_url ของ endpoint ค่าฝุ่นรายสถานี เช่น https://open-api.cmuccdc.org/api/.../{id} (ใช้ {id} หรือ {uri} ได้)"
}

DEFAULT_RT_CONFIG = {
    "_อธิบาย": "แหล่งข้อมูล PM2.5 เรียลไทม์ แก้ไฟล์นี้แล้วรัน app.py ใหม่ (หรือเปิด /api/realtime?refresh=true) | type: air4thai, json",
    "sources": [
        {
            "name": "Air4Thai (กรมควบคุมมลพิษ)",
            "type": "air4thai",
            "enabled": True,
            "url": "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php",
            "match_km": 0,
            "_หมายเหตุ": "จับคู่ด้วยรหัสเท่านั้น: point_id 'PCD86t' <-> stationID '86t' | match_km > 0 จะจับคู่สถานีที่อยู่ใกล้ด้วย (ไม่แนะนำ: เป็นเครื่องวัดคนละเครื่อง)"
        },
        DUSTBOY_SOURCE
    ]
}


def load_rt_config():
    if not os.path.exists(REALTIME_CONFIG_PATH):
        try:
            with open(REALTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_RT_CONFIG, f, ensure_ascii=False, indent=2)
            print("สร้างไฟล์ตั้งค่า", REALTIME_CONFIG_PATH)
        except OSError as e:
            print("เขียนไฟล์ตั้งค่าเรียลไทม์ไม่ได้:", e)
        return DEFAULT_RT_CONFIG
    try:
        with open(REALTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print("[!] อ่าน realtime_sources.json ไม่ได้:", e)
        return {"sources": []}
    srcs = cfg.setdefault("sources", [])
    if not any("cmuccdc" in str(x.get("url", "")) for x in srcs):
        # อัปเกรดไฟล์จากเวอร์ชันก่อน: แทนช่องว่างของเครือข่ายรหัสตัวเลขด้วย DustBoy
        cfg["sources"] = [x for x in srcs if "ใส่-URL" not in str(x.get("url", ""))] + [DUSTBOY_SOURCE]
        try:
            with open(REALTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print("เพิ่ม DustBoy (CMU CCDC) ใน", REALTIME_CONFIG_PATH)
        except OSError as e:
            print("[!] เขียน realtime_sources.json ไม่ได้:", e)
    return cfg


def get_path(obj, path):
    """ดึงค่าจาก JSON ด้วย path คั่นจุด เช่น 'data.0.pm25'"""
    if not path:
        return obj
    for key in str(path).split("."):
        if isinstance(obj, list):
            try:
                obj = obj[int(key)]
            except (ValueError, IndexError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
        if obj is None:
            return None
    return obj


def parse_time(v, fmt=None):
    """แปลงเวลาเป็นเวลาไทย (naive)"""
    if v is None:
        return None
    try:
        if fmt == "epoch" or isinstance(v, (int, float)):
            x = float(v)
            x = x / 1000 if x > 1e12 else x
            return datetime.fromtimestamp(x, TZ_TH).replace(tzinfo=None)
        s = str(v).strip()
        if fmt and fmt != "iso":
            return datetime.strptime(s, fmt)
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return t.astimezone(TZ_TH).replace(tzinfo=None) if t.tzinfo else t
    except (ValueError, OverflowError, OSError):
        return None


def _num(v):
    try:
        x = float(v)
        return x if 0 <= x <= 1000 else None     # Air4Thai ใช้ -1 / "N/A" เมื่อไม่มีค่า
    except (TypeError, ValueError):
        return None


def fetch_air4thai(src):
    """Air4Thai: คืน list ของ (stationID, lat, lon, time, pm25)"""
    r = requests.get(src.get("url", DEFAULT_RT_CONFIG["sources"][0]["url"]), timeout=20)
    r.raise_for_status()
    out = []
    for st in (r.json().get("stations") or []):
        last = st.get("AQILast") or {}
        v = _num((last.get("PM25") or {}).get("value"))
        t = parse_time(f"{last.get('date', '')} {last.get('time', '')}".strip(), "%Y-%m-%d %H:%M")
        if v is None or t is None:
            continue
        try:
            lat, lon = float(st.get("lat")), float(st.get("long"))
        except (TypeError, ValueError):
            lat = lon = None
        out.append((str(st.get("stationID", "")).strip(), lat, lon, t, v))
    return out


def match_air4thai(rows, match_km):
    """จับคู่ค่า Air4Thai กับสถานีของเรา: ด้วยรหัส (PCD86t <-> 86t) ก่อน แล้วด้วยระยะทาง"""
    by_id = {sid.lower(): row for row in rows for sid in [row[0]]}
    found = {}
    for i, pid in enumerate(ST_IDS):
        key = pid[3:].lower() if pid.upper().startswith("PCD") else pid.lower()
        row = by_id.get(key)
        if row is None and match_km > 0:
            near = [(haversine_km(ST_LAT[i], ST_LON[i], r[1], r[2]), r) for r in rows if r[1] is not None]
            near = [x for x in near if x[0] <= match_km]
            row = min(near, key=lambda x: x[0])[1] if near else None
        if row is not None:
            found[pid] = (row[3], row[4])
    return found


# ---------- ตรวจหาช่องข้อมูลอัตโนมัติ (สำหรับ API ที่ยังไม่รู้รูปแบบแน่ชัด) ----------
AUTO_KEYS = {
    "id":    ["id", "dustboy_id", "station_id", "stationid", "dustboy_uri", "uri", "code", "device_id", "sensor_id", "point_id"],
    "value": ["pm25", "pm2_5", "pm2.5", "pm25_value", "log_pm25", "pm25_val", "dustboy_pm25", "pm25_avg", "pm25_1h", "value"],
    "time":  ["log_datetime", "datetime", "date_time", "timestamp", "log_date", "time", "updated_at", "last_update", "created_at", "date"],
    "lat":   ["lat", "latitude", "dustboy_lat", "station_lat"],
    "lon":   ["lon", "lng", "long", "longitude", "dustboy_lon", "dustboy_lng", "station_lon"],
    "uri":   ["dustboy_uri", "uri", "url", "slug"],
}
LIST_HINTS = ["data", "result", "results", "stations", "items", "records", "dustboy", "value"]


def find_list(j):
    """หา list ของสถานีใน JSON (ที่ระดับบนสุด หรือใต้ data/result/stations/...) -> (path, list)"""
    if isinstance(j, list):
        return "", j
    if isinstance(j, dict):
        keys = sorted(j.keys(), key=lambda k: (k.lower() not in LIST_HINTS, k))
        for k in keys:
            v = j[k]
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return k, v
        for k in keys:
            if isinstance(j[k], dict):
                p, lst = find_list(j[k])
                if lst:
                    return f"{k}.{p}".strip("."), lst
    return None, []


def find_key(item, kind, depth=2):
    """หา path ของช่องข้อมูลชนิด kind ใน dict (ไม่สนตัวพิมพ์เล็ก/ใหญ่ ค้นลงไปได้ 2 ชั้น)"""
    if not isinstance(item, dict):
        return None
    low = {k.lower(): k for k in item}
    for cand in AUTO_KEYS[kind]:
        k = low.get(cand)
        if k is None:
            continue
        v = item[k]
        if kind == "value" and isinstance(v, dict):          # เช่น {"pm25": {"value": 12}}
            sub = find_key(v, "value", 0) if depth else None
            if sub:
                return f"{k}.{sub}"
            continue
        if kind == "value" and k.lower() == "value" and _num(v) is None:
            continue
        if v is not None and not isinstance(v, (dict, list)):
            return k
    if depth:
        for k, v in item.items():
            if isinstance(v, dict):
                sub = find_key(v, kind, depth - 1)
                if sub:
                    return f"{k}.{sub}"
    return None


def parse_time_auto(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit()):
        return parse_time(float(v), "epoch")
    for fmt in ("iso", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        t = parse_time(v, fmt)
        if t is not None:
            return t
    return None


def source_headers(src):
    """ใส่ API key ใน header (อ่านจากไฟล์ api_key_file เพื่อไม่ต้องเก็บ key ในไฟล์ตั้งค่า)"""
    key = ""
    kf = src.get("api_key_file")
    if kf:
        p = kf if os.path.isabs(kf) else os.path.join(BASE_DIR, kf)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                key = f.read().strip()
    env_name = src.get("api_key_env") or ("CMUCCDC_API_KEY" if "cmuccdc" in str(src.get("url", "")) else "")
    key = key or (os.environ.get(env_name, "").strip() if env_name else "")
    h = {"User-Agent": "PLK-AirCast/1.0", "Accept": "application/json"}
    for k, v in (src.get("headers") or {}).items():
        h[k] = str(v).replace("{api_key}", key)
    return h, bool(key)


def read_json_items(src):
    """เรียก API แบบรายการเดียว คืน (items, mapping, raw_json)"""
    h, _ = source_headers(src)
    r = requests.get(src["url"], headers=h, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
    j = r.json()
    lp = src.get("list_path", "auto")
    if lp in (None, "", "auto"):
        lp, items = find_list(j)
    else:
        items = get_path(j, lp) or []
    first = items[0] if items else {}
    mapping = {"list_path": lp}
    for kind, cfgkey in (("id", "id_path"), ("value", "value_path"), ("time", "time_path"), ("lat", "lat_path"), ("lon", "lon_path"), ("uri", "uri_path")):
        p = src.get(cfgkey, "auto")
        mapping[kind] = find_key(first, kind) if p in (None, "", "auto") else p
    return items, mapping, j


def latest_record(j, tparse):
    """จากคำตอบของ endpoint ค่าฝุ่น (dict หรือ list ของค่ารายชั่วโมง) คืน (time, value) ล่าสุด"""
    recs = j if isinstance(j, list) else (find_list(j)[1] or [j])
    best = None
    for r in recs:
        if not isinstance(r, dict):
            continue
        vp, tp = find_key(r, "value"), find_key(r, "time")
        v = _num(get_path(r, vp)) if vp else None
        t = tparse(get_path(r, tp)) if tp else None
        if v is None:
            continue
        t = t or datetime.now().replace(second=0, microsecond=0)
        if best is None or t > best[0]:
            best = (t, v)
    return best


def fetch_values_per_station(src, items, mp, rx, tparse):
    """รายชื่อสถานีไม่มีค่าฝุ่น: เรียก value_url ทีละสถานี ({id} / {uri} มาจากรายชื่อสถานี)"""
    h, _ = source_headers(src)
    found, errors = {}, 0
    for it in items:
        sid = str(get_path(it, mp["id"]) or "").strip() if mp["id"] else ""
        if sid not in ST_IDS or not rx.search(sid):
            continue
        uri = str(get_path(it, mp.get("uri")) or "") if mp.get("uri") else ""
        try:
            r = requests.get(src["value_url"].format(id=sid, uri=uri, point_id=sid), headers=h, timeout=12)
            if r.status_code != 200:
                errors += 1
                continue
            rec = latest_record(r.json(), tparse)
        except Exception:
            errors += 1
            continue
        if rec:
            found[sid] = rec
    if not found and errors:
        raise RuntimeError(f"value_url ตอบผิดพลาด {errors} สถานี — ตรวจ URL ใน realtime_sources.json")
    return found


def fetch_json_source(src):
    """แหล่ง JSON ทั่วไป: คืน {point_id: (time, value)}
    จับคู่ด้วยรหัสสถานีก่อน ถ้าไม่ตรงและตั้ง match_km จะจับคู่ด้วยพิกัด (เฉพาะจุดที่แทบทับกัน = เครื่องเดียวกัน)"""
    rx = re.compile(src.get("applies_to") or ".*")
    fmt = src.get("time_format", "auto")
    tparse = parse_time_auto if fmt in (None, "", "auto") else (lambda v: parse_time(v, fmt))
    found = {}
    if src.get("per_station", False):
        h, _ = source_headers(src)
        for pid in ST_IDS:
            if not rx.search(pid):
                continue
            try:
                r = requests.get(src["url"].format(point_id=pid), headers=h, timeout=10)
                if r.status_code != 200:
                    continue
                j = r.json()
            except Exception:
                continue
            vp = src.get("value_path") if src.get("value_path") not in (None, "", "auto") else find_key(j, "value")
            tp = src.get("time_path") if src.get("time_path") not in (None, "", "auto") else find_key(j, "time")
            v, t = _num(get_path(j, vp)), tparse(get_path(j, tp))
            if v is not None and t is not None:
                found[pid] = (t, v)
        return found

    items, mp, _ = read_json_items(src)
    if not items:
        raise RuntimeError("ไม่พบรายการสถานีในข้อมูลที่ได้ (ดู /api/realtime/probe)")
    if not mp["value"]:
        if src.get("value_url"):
            return fetch_values_per_station(src, items, mp, rx, tparse)
        raise RuntimeError(f"รายการสถานีไม่มีช่องค่าฝุ่น (ช่องที่มี: {', '.join(list(items[0].keys())[:12])}) — "
                           "ใส่ value_url ใน realtime_sources.json (ดู /api/realtime/probe)")
    match_km = float(src.get("match_km", 0))
    targets = [i for i, pid in enumerate(ST_IDS) if rx.search(pid)]
    by_id = {pid: i for i, pid in enumerate(ST_IDS)}
    for it in items:
        v = _num(get_path(it, mp["value"]))
        t = tparse(get_path(it, mp["time"])) if mp["time"] else None
        if v is None:
            continue
        if t is None:
            t = datetime.now().replace(second=0, microsecond=0)   # ไม่มีเวลาในข้อมูล: ใช้เวลาที่ดึง
        pid = str(get_path(it, mp["id"]) or "").strip() if mp["id"] else ""
        i = by_id.get(pid)
        if (i is None or i not in targets) and match_km > 0 and mp["lat"] and mp["lon"]:
            try:
                la, lo = float(get_path(it, mp["lat"])), float(get_path(it, mp["lon"]))
                d = haversine_km(la, lo, ST_LAT[targets], ST_LON[targets])
                k = int(np.argmin(d))
                i = targets[k] if d[k] <= match_km else None
            except (TypeError, ValueError):
                i = None
        if i is not None and i in targets:
            prev = found.get(ST_IDS[i])
            if prev is None or t > prev[0]:
                found[ST_IDS[i]] = (t, v)
    return found


def poll_realtime():
    """ดึงค่าเรียลไทม์จากทุกแหล่งที่เปิดใช้ แล้วบันทึกลง realtime_log.csv"""
    cfg = load_rt_config()
    status, new_rows = [], []
    for src in cfg.get("sources", []):
        if not src.get("enabled"):
            continue
        name = src.get("name", src.get("type"))
        try:
            if src.get("type") == "air4thai":
                found = match_air4thai(fetch_air4thai(src), float(src.get("match_km", 0)))
            elif src.get("type") == "json":
                if "ใส่-URL" in src.get("url", ""):
                    status.append({"name": name, "ok": False, "stations": 0, "message": "ยังไม่ได้ใส่ URL"}); continue
                if src.get("api_key_file") and not source_headers(src)[1]:
                    status.append({"name": name, "ok": False, "stations": 0, "message": f"ไม่พบ API key (ไฟล์ {src['api_key_file']} หรือ Secret CMUCCDC_API_KEY)"}); continue
                found = fetch_json_source(src)
            else:
                status.append({"name": name, "ok": False, "stations": 0, "message": "ไม่รู้จัก type"}); continue
            for pid, (t, v) in found.items():
                new_rows.append((pid, t, v, name))
            status.append({"name": name, "ok": True, "stations": len(found), "message": ""})
        except Exception as e:
            status.append({"name": name, "ok": False, "stations": 0, "message": f"{type(e).__name__}: {e}"[:200]})
    with _rt_lock:
        existing = read_realtime_log()
        write_header = not os.path.exists(REALTIME_LOG_PATH)
        try:
            with open(REALTIME_LOG_PATH, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["point_id", "time", "pm25", "source", "fetched_at"])
                now_s = datetime.now().strftime("%Y-%m-%d %H:%M")
                for pid, t, v, name in new_rows:
                    if existing.get(pid, {}).get(t.replace(minute=0)) == v:
                        continue                     # ค่าเดิมของชั่วโมงเดิม ไม่ต้องบันทึกซ้ำ
                    w.writerow([pid, t.strftime("%Y-%m-%d %H:%M"), round(v, 2), name, now_s])
        except OSError as e:
            status.append({"name": "บันทึก realtime_log.csv", "ok": False, "stations": 0, "message": str(e)})
        for pid, t, v, name in new_rows:
            prev = RT_LATEST.get(pid)
            if prev is None or t >= prev["time"]:
                RT_LATEST[pid] = {"value": v, "time": t, "source": name}
        RT_STATUS["last_poll"] = datetime.now()
        RT_STATUS["sources"] = status
    return status


def realtime_poller():
    while True:
        try:
            poll_realtime()
            cached_run["timestamp"] = None          # ให้คำนวณผลใหม่ด้วยข้อมูลล่าสุด
            cached_grids.clear()
        except Exception:
            traceback.print_exc()
        threading.Event().wait(REALTIME_POLL_MIN * 60)


def realtime_now():
    """ค่าเรียลไทม์ที่ยังสด (ไม่เกิน REALTIME_FRESH_HOURS ชั่วโมง)"""
    now = datetime.now()
    return {pid: r for pid, r in RT_LATEST.items() if (now - r["time"]).total_seconds() <= REALTIME_FRESH_HOURS * 3600}


# =============================================================================
# กริด 1 x 1 กม.
# =============================================================================
MIN_LAT, MAX_LAT = 16.30, 17.42
MIN_LON, MAX_LON = 99.92, 100.90
CITY = (16.8211, 100.2659)
COS_LAT = math.cos(math.radians(16.85))
GRID_KM = 1.0
DLAT = GRID_KM / 110.574
DLON = GRID_KM / (111.320 * COS_LAT)
G_LATS = np.arange(MIN_LAT, MAX_LAT + 1e-9, DLAT)
G_LONS = np.arange(MIN_LON, MAX_LON + 1e-9, DLON)
NY, NX = len(G_LATS), len(G_LONS)
G_LAT = np.repeat(G_LATS, NX)          # lat-major: index = iy * NX + ix (iy จากใต้ไปเหนือ)
G_LON = np.tile(G_LONS, NY)
GRID_BOUNDS = [float(G_LONS[0]), float(G_LATS[0]), float(G_LONS[-1]), float(G_LATS[-1])]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def kernel_weights(src_lat, src_lon, dst_lat, dst_lon, bandwidth_deg=0.16):
    """Gaussian kernel (normalize แล้ว): ผิวเรียบ ไม่ overshoot เกินค่าต้นทาง"""
    dy = dst_lat[:, None] - src_lat[None, :]
    dx = (dst_lon[:, None] - src_lon[None, :]) * COS_LAT
    w = np.exp(-(dx ** 2 + dy ** 2) / (2 * bandwidth_deg ** 2))
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)


def idw_weights(src_lat, src_lon, dst_lat, dst_lon, power=2.0):
    if not len(src_lat):
        return np.zeros((len(dst_lat), 0))
    d2 = (dst_lat[:, None] - src_lat[None, :]) ** 2 + ((dst_lon[:, None] - src_lon[None, :]) * COS_LAT) ** 2
    w = 1.0 / (d2 ** (power / 2) + 1e-4)
    return w / w.sum(axis=1, keepdims=True)


WX_LAT = np.repeat(np.linspace(16.22, 17.50, 5), 5)    # 25 จุด (ระยะห่าง ~0.3° ใกล้เคียงกริด ERA5 0.25°)
WX_LON = np.tile(np.linspace(99.84, 100.98, 5), 5)
N_WX = len(WX_LAT)
K_GRID = kernel_weights(WX_LAT, WX_LON, G_LAT, G_LON)
K_ST = kernel_weights(WX_LAT, WX_LON, ST_LAT, ST_LON) if N_ST else np.zeros((0, N_WX))
IDW_GRID = idw_weights(ST_LAT, ST_LON, G_LAT, G_LON)
UHI_SHAPE = np.exp(-((G_LAT - CITY[0]) ** 2 + ((G_LON - CITY[1]) * COS_LAT) ** 2) / (2 * 0.07 ** 2))


def now_hour():
    return datetime.now().replace(minute=0, second=0, microsecond=0)


def rollout_start(today):
    """วันแรกที่ต้องให้โมเดลคำนวณ = วันถัดจากค่าตรวจวัดล่าสุด (ไม่เกิน 85 วันย้อนหลัง)"""
    first_needed = today - timedelta(days=HISTORY_HOURS // 24 + 1)
    s = (DATA_AS_OF + timedelta(days=1)) if DATA_AS_OF else first_needed
    return max(min(s, first_needed), today - timedelta(days=MAX_ROLLOUT_DAYS))


# =============================================================================
# อุตุนิยมวิทยา (Open-Meteo): รายชั่วโมง 26 พิกัด ตั้งแต่วันเริ่ม rollout ถึง +3 วัน
# =============================================================================
cached_meteo = {"timestamp": None, "data": None}
VAR_SETS = [
    {"ws": "wind_speed_10m", "wd": "wind_direction_10m", "mh": "boundary_layer_height",
     "t2m": "temperature_2m", "rh": "relative_humidity_2m", "lst": "surface_temperature"},
    {"ws": "wind_speed_10m", "wd": "wind_direction_10m", "mh": "boundary_layer_height",
     "t2m": "temperature_2m", "rh": "relative_humidity_2m", "lst": "soil_temperature_0cm"},
    {"ws": "wind_speed_10m", "wd": "wind_direction_10m",
     "t2m": "temperature_2m", "rh": "relative_humidity_2m"},
]
DEFAULTS = {"ws": 1.5, "wd": 200.0, "mh": 450.0, "t2m": 28.0, "rh": 65.0, "lst": 27.0}


def _fill(a, default):
    a = np.array(a, float)
    for j in range(a.shape[1]):
        col = a[:, j]
        ok = ~np.isnan(col)
        a[:, j] = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok]) if ok.any() else default
    return a


def _simulated_meteo(t0, past_days):
    start = datetime.combine(t0.date() - timedelta(days=past_days), datetime.min.time())
    T = (past_days + 4) * 24
    times = [start + timedelta(hours=i) for i in range(T)]
    hrs = np.array([t.hour for t in times], float)[:, None]
    i = np.arange(T, dtype=float)[:, None]
    lat = np.append(WX_LAT, CITY[0])[None, :]
    lon = np.append(WX_LON, CITY[1])[None, :]
    diurnal = np.sin((hrs - 8) * np.pi / 12)
    return {
        "source": "simulated", "times": times, "now_idx": times.index(t0), "missing": [],
        "ws": np.maximum(0.6, (1.8 + 0.8 * np.sin(i * 0.15)) * (1 - 0.25 * (lon - 100.0) / 0.9)),
        "wd": (210 + 30 * np.sin(i * 0.05) + (lat - 16.8) * 15) % 360,
        "mh": np.maximum(120, 450 + 380 * diurnal) * (1 + 0.15 * np.sin((lat - 16.4) * 3.2)),
        "t2m": 28 + 5 * diurnal + 0 * lat, "rh": 65 - 20 * diurnal + 0 * lat, "lst": 27 + 7 * diurnal + 0 * lat,
    }


def fetch_meteo():
    """dict: times (datetime เวลาไทย), now_idx, และ array (T, 26) — คอลัมน์ 0..24 = กริด, 25 = ตัวเมือง"""
    t0 = now_hour()
    c = cached_meteo
    if c["data"] and c["timestamp"] and (datetime.now() - c["timestamp"]).total_seconds() < 1800:
        return c["data"]
    past_days = min(92, (t0.date() - rollout_start(t0.date())).days + 1)
    lats = list(WX_LAT) + [CITY[0]]
    lons = list(WX_LON) + [CITY[1]]
    for vs in VAR_SETS:
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in lats),
            "longitude": ",".join(f"{v:.4f}" for v in lons),
            "hourly": ",".join(vs.values()),
            "wind_speed_unit": "ms", "timezone": "Asia/Bangkok",
            "past_days": past_days, "forecast_days": 4,
        }
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
        except Exception as e:
            print("Open-Meteo connection error:", e)
            break
        if r.status_code == 400:
            print("Open-Meteo rejected variables", list(vs.values()), "->", r.text[:160])
            continue
        if r.status_code != 200:
            print("Open-Meteo HTTP", r.status_code, r.text[:160])
            break
        payload = r.json()
        payload = payload if isinstance(payload, list) else [payload]
        if len(payload) != len(lats):
            print("Open-Meteo: unexpected location count", len(payload))
            break
        tstr = payload[0]["hourly"]["time"]
        key = t0.strftime("%Y-%m-%dT%H:00")
        if key not in tstr:
            print("Open-Meteo: current hour not in response")
            break
        data = {"source": "open-meteo", "times": [datetime.strptime(s, "%Y-%m-%dT%H:%M") for s in tstr],
                "now_idx": tstr.index(key), "missing": [k for k in DEFAULTS if k not in vs]}
        for k in DEFAULTS:
            if k in vs:
                cols = [[(np.nan if v is None else v) for v in loc["hourly"].get(vs[k], [])] for loc in payload]
                n = min(len(x) for x in cols)
                arr = np.array([x[:n] for x in cols], float).T
                if np.isnan(arr).all():
                    data["missing"].append(k)
                data[k] = _fill(arr, DEFAULTS[k])
            else:
                data[k] = np.full((len(tstr), len(lats)), DEFAULTS[k])
        c["data"], c["timestamp"] = data, datetime.now()
        return data
    return _simulated_meteo(t0, past_days)


def meteo_idx(m, offset):
    return int(np.clip(m["now_idx"] + offset, 0, m["ws"].shape[0] - 1))


def uv_from(ws, wd):
    rad = np.radians(wd)
    return -ws * np.sin(rad), -ws * np.cos(rad)


def weather_at(m, offset, weights):
    """ค่ารายชั่วโมง (ใช้แสดงผล): ลม u, v, ความเร็ว, ทิศ, ความสูงชั้นผสม"""
    i = meteo_idx(m, offset)
    u, v = uv_from(m["ws"][i, :N_WX], m["wd"][i, :N_WX])
    ui, vi, mh = weights @ u, weights @ v, weights @ m["mh"][i, :N_WX]
    return ui, vi, np.hypot(ui, vi), (np.degrees(np.arctan2(-ui, -vi)) + 360) % 360, mh


def city_weather(m, offset):
    i = meteo_idx(m, offset)
    return {"wind_speed": float(m["ws"][i, N_WX]), "wind_dir": float(m["wd"][i, N_WX]),
            "mixing_height": float(m["mh"][i, N_WX]), "temp": float(m["t2m"][i, N_WX]),
            "humidity": float(m["rh"][i, N_WX]), "surface_temp": float(m["lst"][i, N_WX])}


def daily_station_weather(m, d):
    """ค่าอากาศ "เฉลี่ยรายวัน" ของแต่ละสถานี (วันตามเวลาไทย) — แบบเดียวกับข้อมูล ERA5 รายวันที่ใช้ฝึก
    ลม: เฉลี่ย u, v ก่อนแล้วจึงคำนวณความเร็วและทิศ (vector mean)"""
    idx = [i for i, t in enumerate(m["times"]) if t.date() == d]
    if not idx:      # นอกช่วงข้อมูล: ใช้วันที่ใกล้ที่สุด
        nearest = min(range(len(m["times"])), key=lambda i: abs((m["times"][i].date() - d).days))
        idx = [i for i, t in enumerate(m["times"]) if t.date() == m["times"][nearest].date()]
    u, v = uv_from(m["ws"][idx, :N_WX], m["wd"][idx, :N_WX])        # (h, 25)
    us, vs = (u.mean(axis=0) @ K_ST.T), (v.mean(axis=0) @ K_ST.T)    # (stations,)
    mh = m["mh"][idx, :N_WX].mean(axis=0) @ K_ST.T
    ws = np.hypot(us, vs)
    wd = (np.degrees(np.arctan2(-us, -vs)) + 360) % 360
    return ws, wd, mh, len(idx)


# =============================================================================
# จุดความร้อน NASA FIRMS — ค่าเริ่มต้น MODIS ให้ตรงกับ ee.ImageCollection("FIRMS") ที่ใช้ตอนฝึก
# =============================================================================
FIRMS_SOURCES = [s.strip() for s in os.environ.get("FIRMS_SOURCES", "MODIS_NRT").split(",") if s.strip()]
FIRMS_BBOX = (MIN_LON - 0.55, MIN_LAT - 0.5, MAX_LON + 0.55, MAX_LAT + 0.5)
cached_firms = {"timestamp": None, "data": None}


def firms_key():
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key and os.path.exists(FIRMS_KEY_PATH):
        with open(FIRMS_KEY_PATH, "r", encoding="utf-8") as f:
            key = f.read().strip()
    return key


def _parse_firms_csv(text, source):
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            t = datetime.strptime(f"{row['acq_date']} {row['acq_time'].zfill(4)}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        raw = (row.get("confidence") or "").strip().lower()
        conf = {"l": "ต่ำ", "n": "ปกติ", "h": "สูง"}.get(raw)
        if conf is None:
            try:
                cv = int(float(raw)); conf = "สูง" if cv >= 80 else ("ปกติ" if cv >= 30 else "ต่ำ")
            except ValueError:
                conf = "-"
        try:
            frp = float(row.get("frp") or 0)
        except ValueError:
            frp = 0.0
        out.append({"lat": lat, "lon": lon, "frp_mw": frp, "confidence": conf,
                    "satellite": row.get("satellite") or source, "source": source,
                    "daynight": row.get("daynight", ""), "time_utc": t})
    return out


def fetch_firms():
    """ดึงจุดความร้อนตั้งแต่ (วันเริ่ม rollout − 7 วัน) ถึงวันนี้ เพื่อคำนวณ lag และผลรวม 7 วันได้ครบ"""
    now = datetime.now(timezone.utc)
    c = cached_firms
    if c["data"] and c["timestamp"] and (now - c["timestamp"]).total_seconds() < 1800:
        return c["data"]
    key = firms_key()
    if not key:
        return {"status": "no_key", "points": [],
                "message": "ยังไม่ได้ตั้งค่า NASA FIRMS MAP_KEY (ไฟล์ firms_key.txt หรือ env FIRMS_MAP_KEY)"}
    w, s, e, n = FIRMS_BBOX
    today = now.astimezone(TZ_TH).date()
    first = rollout_start(today) - timedelta(days=8)
    chunks, start = [], first
    while start <= today:
        days = min(5, (today - start).days + 1)         # API ให้ช่วงละไม่เกิน 5 วัน
        chunks.append((start, days))
        start += timedelta(days=days)
    points, errors, seen = [], [], set()
    for src in FIRMS_SOURCES:
        for d0, days in chunks:
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/"
                   f"{w:.3f},{s:.3f},{e:.3f},{n:.3f}/{days}/{d0.isoformat()}")
            try:
                r = requests.get(url, timeout=25)
                text = r.text.strip()
                if r.status_code != 200 or not text.lower().startswith("latitude"):
                    errors.append(f"{src} {d0}: {text[:100] or r.status_code}")
                    continue
                for p in _parse_firms_csv(text, src):
                    k = (round(p["lat"], 4), round(p["lon"], 4), p["time_utc"], src)
                    if k not in seen:
                        seen.add(k); points.append(p)
            except Exception as ex:
                errors.append(f"{src} {d0}: {ex}")
    if not points and errors and len(errors) == len(FIRMS_SOURCES) * len(chunks):
        result = {"status": "error", "points": [], "message": "; ".join(errors[:3])}
        c.update(timestamp=now - timedelta(seconds=1500), data=result)
        return result
    result = {"status": "ok", "points": points, "message": "; ".join(errors[:3]),
              "sources": FIRMS_SOURCES, "fetched_at": now.astimezone(TZ_TH).strftime("%Y-%m-%d %H:%M")}
    c.update(timestamp=now, data=result)
    return result


# =============================================================================
# จุดความร้อน GISTDA (VIIRS 30 วัน) — ใช้แทน FIRMS เมื่อมีไฟล์ gistda_key.txt
# =============================================================================
GISTDA_URL = "https://api-gateway.gistda.or.th/api/2.0/resources/features/viirs/30days"
GISTDA_PAGE = 1000
PROVINCE_TH = "พิษณุโลก"


def gistda_key():
    k = os.environ.get("GISTDA_API_KEY", "").strip()
    if not k and os.path.exists(GISTDA_KEY_PATH):
        with open(GISTDA_KEY_PATH, "r", encoding="utf-8") as f:
            k = f.read().strip()
    return k


def _ci(d, *names):
    """อ่านค่าจาก dict แบบไม่สนตัวพิมพ์ ตามชื่อที่ให้ตามลำดับ"""
    if not isinstance(d, dict):
        return None
    low = {k.lower(): k for k in d}
    for n in names:
        k = low.get(n.lower())
        if k is not None and d[k] not in (None, ""):
            return d[k]
    return None


def _gistda_time(p):
    """เวลาตรวจพบ -> datetime UTC
    ถ้ามี th_date/th_time (เวลาไทย) ใช้ก่อน, ไม่งั้นใช้ acq_date/acq_time ซึ่งตามมาตรฐาน FIRMS เป็นเวลา UTC"""
    for dkey, tkey, tz in (("th_date", "th_time", TZ_TH), ("acq_date", "acq_time", timezone.utc)):
        dv, tv = _ci(p, dkey), _ci(p, tkey)
        if dv is None:
            continue
        ds = str(dv).strip()[:10]
        ts = str(tv if tv is not None else "0000").strip().replace(":", "")
        ts = ts.split(".")[0].zfill(4)[:4] if ts.replace("0", "") or ts else "0000"
        try:
            return datetime.strptime(f"{ds} {ts}", "%Y-%m-%d %H%M").replace(tzinfo=tz).astimezone(timezone.utc), dkey
        except ValueError:
            try:   # บางระบบให้ datetime เต็มในช่องวันที่
                t = datetime.fromisoformat(str(dv).replace("Z", "+00:00"))
                return (t if t.tzinfo else t.replace(tzinfo=tz)).astimezone(timezone.utc), dkey
            except ValueError:
                continue
    return None, None


def parse_gistda_features(j):
    """GeoJSON / JSON จาก GISTDA -> list ของจุด (รูปแบบเดียวกับ FIRMS) และข้อมูลการตรวจหาช่อง"""
    feats = j.get("features") if isinstance(j, dict) else None
    if feats is None:
        _, feats = find_list(j)
    out, info = [], {"time_field": None, "province_field": None}
    for f in feats or []:
        p = f.get("properties", f) if isinstance(f, dict) else {}
        geom = f.get("geometry") if isinstance(f, dict) else None
        lat, lon = _ci(p, "latitude", "lat"), _ci(p, "longitude", "lon", "long")
        if (lat is None or lon is None) and isinstance(geom, dict) and isinstance(geom.get("coordinates"), list):
            lon, lat = geom["coordinates"][:2]
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        t, tf = _gistda_time(p)
        if t is None:
            continue
        info["time_field"] = info["time_field"] or tf
        prov = _ci(p, "pv_tn", "province", "prov_tn", "pv_th", "changwat")
        if prov is not None:
            info["province_field"] = info["province_field"] or "pv_tn/province"
        raw = str(_ci(p, "confidence") or "").strip().lower()
        conf = {"l": "ต่ำ", "low": "ต่ำ", "n": "ปกติ", "nominal": "ปกติ", "h": "สูง", "high": "สูง"}.get(raw, raw or "-")
        try:
            frp = float(_ci(p, "frp") or 0)
        except (TypeError, ValueError):
            frp = 0.0
        out.append({"lat": lat, "lon": lon, "frp_mw": frp, "confidence": conf,
                    "satellite": str(_ci(p, "satellite", "sat") or "VIIRS"), "source": "GISTDA VIIRS",
                    "daynight": str(_ci(p, "daynight") or ""), "time_utc": t,
                    "in_prov": (PROVINCE_TH in str(prov)) if prov is not None else None,
                    "district_th": _ci(p, "ap_tn", "amphoe", "district")})
    return out, info


def fetch_gistda(key):
    """ดึงจุดความร้อน 30 วันในกรอบจังหวัด+กันชน แบบแบ่งหน้า (limit/offset)
    อ่านต่อจนครบ numberMatched (เซิร์ฟเวอร์อาจคืนน้อยกว่า limit ที่ขอต่อหน้า)"""
    w, s, e, n = FIRMS_BBOX
    headers = {"accept": "application/json", "API-Key": key, "User-Agent": "PLK-AirCast/1.0"}
    points, info, offset, seen = [], {}, 0, set()
    for _ in range(60):                                    # กันวนไม่รู้จบ (สูงสุด 60 หน้า)
        params = {"bbox": f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}", "limit": GISTDA_PAGE, "offset": offset,
                  "ct_tn": "ราชอาณาจักรไทย"}
        r = requests.get(GISTDA_URL, params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"GISTDA HTTP {r.status_code}: {r.text[:150]}")
        j = r.json()
        raw = j.get("features") if isinstance(j, dict) and "features" in j else (find_list(j)[1] or [])
        batch, inf = parse_gistda_features({"features": raw})
        info = {k: info.get(k) or inf.get(k) for k in ("time_field", "province_field")}
        for p, f in zip(batch, raw):
            pid = (f.get("properties") or {}).get("hotspotid") or f.get("id") or (p["lat"], p["lon"], p["time_utc"])
            if pid not in seen:
                seen.add(pid); points.append(p)
        got = int(_ci(j, "numberReturned") or len(raw)) if isinstance(j, dict) else len(raw)
        total = _ci(j, "numberMatched", "total", "count") if isinstance(j, dict) else None
        offset += got
        if got == 0 or (isinstance(total, (int, float)) and offset >= total) or (total is None and got < GISTDA_PAGE):
            break
    info["pages_read_to_offset"] = offset
    return points, info


def fetch_hotspots():
    """เลือกแหล่งจุดความร้อน: GISTDA (ถ้ามี gistda_key.txt) ไม่งั้น NASA FIRMS"""
    key = gistda_key()
    if not key:
        return fetch_firms()
    now = datetime.now(timezone.utc)
    c = cached_firms
    if c["data"] and c["timestamp"] and c["data"].get("provider") == "gistda" and (now - c["timestamp"]).total_seconds() < 1800:
        return c["data"]
    try:
        points, info = fetch_gistda(key)
        result = {"status": "ok", "provider": "gistda", "points": points, "message": "",
                  "sources": ["GISTDA VIIRS"], "fields": info,
                  "fetched_at": now.astimezone(TZ_TH).strftime("%Y-%m-%d %H:%M")}
        c.update(timestamp=now, data=result)
    except Exception as e:
        result = {"status": "error", "provider": "gistda", "points": [], "message": f"{type(e).__name__}: {e}"[:250]}
        c.update(timestamp=now - timedelta(seconds=1500), data=result)       # ลองใหม่ใน 5 นาที
    return result


def hotspot_label():
    return "GISTDA VIIRS" if gistda_key() else "FIRMS " + "/".join(FIRMS_SOURCES).replace("_NRT", "")


def in_province(lat, lon):
    return (lat >= MIN_LAT) & (lat <= MAX_LAT) & (lon >= MIN_LON) & (lon <= MAX_LON)


DEFAULT_HOTSPOT = {"h10k": 0, "h25k": 2, "h50k": 5, "h_prov": 9}
HS_FEATURES = ['hotspot_10km_count', 'hotspot_25km_count', 'hotspot_50km_count', 'hotspot_province_count',
               'hotspot_local_ratio', 'hotspot_25km_count_lag_1', 'hotspot_25km_count_lag_2',
               'hotspot_province_count_lag_1', 'hotspot_province_count_lag_2', 'hotspot_province_count_lag_3',
               'hotspot_province_roll_sum_3d', 'hotspot_province_roll_sum_7d']


class HotspotIndex:
    """นับจุดความร้อน "ตามวันปฏิทินเวลาไทย" (แบบ filterDate รายวันตอนฝึก) รอบสถานี 10/25/50 กม. และทั้งจังหวัด"""

    def __init__(self, firms):
        self.ok = firms.get("status") == "ok"
        pts = firms.get("points", []) if self.ok else []
        self.day = np.array([p["time_utc"].astimezone(TZ_TH).date().toordinal() for p in pts], int)
        lat = np.array([p["lat"] for p in pts]); lon = np.array([p["lon"] for p in pts])
        bbox_prov = in_province(lat, lon) if len(pts) else np.zeros(0, bool)
        # ถ้าแหล่งข้อมูลระบุจังหวัดมา (GISTDA pv_tn) ใช้ชื่อจังหวัด ไม่งั้นใช้กรอบสี่เหลี่ยม
        self.prov = np.array([p["in_prov"] if p.get("in_prov") is not None else b for p, b in zip(pts, bbox_prov)], float)
        d = haversine_km(ST_LAT[:, None], ST_LON[:, None], lat[None, :], lon[None, :]) if (len(pts) and N_ST) else np.zeros((N_ST, 0))
        # ต้องเป็นตัวเลข: bool @ bool ใน numpy ให้ผลเป็น OR ไม่ใช่การนับ
        self.m10, self.m25, self.m50 = (d <= 10).astype(float), (d <= 25).astype(float), (d <= 50).astype(float)
        self.today = datetime.now(TZ_TH).date()

    def _on(self, d):
        return (self.day == d.toordinal()).astype(float)

    def province_count(self, d):
        return int(self.prov @ self._on(d)) if len(self.day) else 0

    def features(self, d):
        """feature จุดความร้อนของวัน d (array n_stations x 12) — วันในอนาคตใช้ค่าของวันนี้ (persistence)"""
        if not self.ok:
            k = DEFAULT_HOTSPOT
            row = [k["h10k"], k["h25k"], k["h50k"], k["h_prov"], (k["h10k"] + 1) / (k["h50k"] + 1),
                   k["h25k"], k["h25k"], k["h_prov"], k["h_prov"], k["h_prov"], k["h_prov"] * 2.5, k["h_prov"] * 5.0]
            return np.tile(row, (N_ST, 1)).astype(float)
        d = min(d, self.today)
        D = lambda k: self._on(d - timedelta(days=k))
        p = [self.province_count(d - timedelta(days=k)) for k in range(8)]
        c10, c25_0, c50 = self.m10 @ D(0), self.m25 @ D(0), self.m50 @ D(0)
        out = np.zeros((N_ST, 12))
        out[:, 0], out[:, 1], out[:, 2], out[:, 3] = c10, c25_0, c50, p[0]
        out[:, 4] = (c10 + 1) / (c50 + 1)
        out[:, 5], out[:, 6] = self.m25 @ D(1), self.m25 @ D(2)
        out[:, 7], out[:, 8], out[:, 9] = p[1], p[2], p[3]
        out[:, 10] = p[1] + p[2] + p[3]
        out[:, 11] = sum(p[1:8])
        return out


# =============================================================================
# โมเดลรายวัน + rollout แบบ recursive
# =============================================================================
def initial_lag_state(start):
    """ค่า PM2.5 ย้อนหลังของแต่ละสถานีก่อนวันเริ่ม rollout
    สถานีที่ข้อมูลเก่าเกิน STALE_DAYS ใช้ค่าเฉลี่ยจังหวัดของวันเดียวกันแทน (ระบุว่าเป็นค่าแทน)"""
    hist, source = [], []
    prov_dates = sorted(d for d in PROV_MEAN_OBS if d < start)
    prov_recent = [PROV_MEAN_OBS[d] for d in prov_dates[-7:]] or [20.0] * 7
    for pid in ST_IDS:
        last = LAST_OBS.get(pid)
        if last and DATA_AS_OF and (DATA_AS_OF - last).days <= STALE_DAYS:
            vals = [OBS[pid][d] for d in sorted(OBS[pid]) if d < start][-7:]
            hist.append(list(vals)); source.append("obs")
        else:
            hist.append(list(prov_recent)); source.append("province")
    for h in hist:                       # เติมให้ครบ 7 ค่า
        while len(h) < 7:
            h.insert(0, h[0] if h else 20.0)
    return hist, source


def lag_features(h):
    l1, l2, l3, l7 = h[-1], h[-2], h[-3], h[-7]
    return {'pm25_lag_1': l1, 'pm25_lag_2': l2, 'pm25_lag_3': l3, 'pm25_lag_7': l7,
            'pm25_roll_mean_3d': float(np.mean(h[-3:])), 'pm25_roll_std_3d': float(np.std(h[-3:], ddof=1)),
            'pm25_roll_mean_7d': float(np.mean(h[-7:])), 'pm25_roll_max_7d': float(np.max(h[-7:]))}


def run_daily_model(m, hsi):
    """คืน dict วันที่ -> {"values": array(stations), "source": list('obs'|'model'|'province')}
    ครอบคลุมตั้งแต่ 7 วันก่อนถึง +3 วัน"""
    today = datetime.now(TZ_TH).date()
    start = rollout_start(today)
    end = today + timedelta(days=3)
    hist, lag_src = initial_lag_state(start)
    out = {}
    first_shown = today - timedelta(days=HISTORY_HOURS // 24 + 1)

    # วันในช่วงแสดงผลที่อยู่ก่อนวันเริ่ม rollout: ใช้ค่าตรวจวัดจริง
    d = first_shown
    while d < start:
        vals = np.array([OBS[pid].get(d, np.nan) for pid in ST_IDS])
        src = ["obs" if np.isfinite(v) else "province" for v in vals]
        vals = np.where(np.isfinite(vals), vals, PROV_MEAN_OBS.get(d, np.nanmean(vals) if np.isfinite(vals).any() else 20.0))
        out[d] = {"values": vals, "source": src}
        d += timedelta(days=1)

    if not N_ST:
        return out, {"start": start, "lag_source": lag_src}
    feats = model_data["features"] if model_data else None
    col = {f: j for j, f in enumerate(feats)} if feats else {}
    d = start
    prev_prov = PROV_MEAN_OBS.get(start - timedelta(days=1), float(np.mean([h[-1] for h in hist])))
    while d <= end:
        obs_today = np.array([OBS[pid].get(d, np.nan) for pid in ST_IDS])
        if model_data:
            ws, wd, mh, _ = daily_station_weather(m, d)
            hs = hsi.features(d)
            X = np.zeros((N_ST, len(feats)))
            doy = d.timetuple().tm_yday
            rad = np.radians(wd)
            vent = ws * mh
            common = {
                'sin_doy': math.sin(2 * math.pi * doy / 365.25), 'cos_doy': math.cos(2 * math.pi * doy / 365.25),
                'is_burning_season': 1 if d.month in (1, 2, 3, 4) else 0, 'month': d.month,
                'dayofweek': d.weekday(), 'is_weekend': 1 if d.weekday() >= 5 else 0,
                'lst_missing': 1, 'prov_pm25_mean_lag_1': prev_prov,
            }
            per = {'lat': ST_LAT, 'lon': ST_LON, 'wind_speed_station_m_s': ws,
                   'wind_u': -ws * np.sin(rad), 'wind_v': -ws * np.cos(rad), 'mixing_height_station_m': mh,
                   'ventilation_idx': vent, 'stagnation_idx': 1.0 / (vent + 1e-3),
                   'lst_imputed': np.array([LST_LAST[pid] for pid in ST_IDS])}
            for f, v in {**common, **per}.items():
                if f in col:
                    X[:, col[f]] = v
            for j, f in enumerate(HS_FEATURES):
                if f in col:
                    X[:, col[f]] = hs[:, j]
            for i in range(N_ST):
                for f, v in lag_features(hist[i]).items():
                    if f in col:
                        X[i, col[f]] = v
            pred = np.maximum(1.0, model_data["model"].predict(X))
        else:
            pred = np.array([h[-1] for h in hist])
        # ถ้ามีค่าตรวจวัดของวันนั้น ใช้ค่าจริงแทน (ทั้งแสดงผลและเป็น lag ของวันถัดไป)
        vals = np.where(np.isfinite(obs_today), obs_today, pred)
        src = ["obs" if np.isfinite(o) else "model" for o in obs_today]
        out[d] = {"values": vals, "source": src}
        for i in range(N_ST):
            hist[i].append(float(vals[i])); hist[i] = hist[i][-7:]
        prev_prov = float(np.mean(vals))
        d += timedelta(days=1)
    return out, {"start": start, "lag_source": lag_src}


cached_run = {"timestamp": None, "data": None}


def get_engine():
    c = cached_run
    if c["data"] and c["timestamp"] and (datetime.now() - c["timestamp"]).total_seconds() < 900:
        return c["data"]
    problems = []
    try:
        refresh_observations()          # รวมค่าเฉลี่ยรายวันที่สะสมจากระบบเรียลไทม์
    except Exception as e:
        traceback.print_exc()
        problems.append(f"ข้อมูลเรียลไทม์สะสม: {type(e).__name__}: {e}")
    try:
        m = fetch_meteo()
    except Exception as e:
        traceback.print_exc()
        problems.append(f"อากาศ: {type(e).__name__}: {e}")
        m = _simulated_meteo(now_hour(), min(92, (now_hour().date() - rollout_start(now_hour().date())).days + 1))
    try:
        firms = fetch_hotspots()
    except Exception as e:
        traceback.print_exc()
        problems.append(f"จุดความร้อน: {type(e).__name__}: {e}")
        firms = {"status": "error", "points": [], "message": str(e)}
    hsi = HotspotIndex(firms)
    try:
        daily, info = run_daily_model(m, hsi)
    except Exception as e:
        # โมเดลคำนวณไม่ได้: แสดงค่าตรวจวัดล่าสุดของแต่ละสถานีแทน (persistence) พร้อมแจ้งเตือน
        traceback.print_exc()
        problems.append(f"โมเดล: {type(e).__name__}: {e}")
        global model_data
        saved, model_data = model_data, None
        try:
            daily, info = run_daily_model(m, hsi)
        finally:
            model_data = saved
    info["problems"] = problems
    eng = {"t0": now_hour(), "meteo": m, "firms": firms, "hsi": hsi, "daily": daily, "info": info}
    c.update(timestamp=datetime.now(), data=eng)
    return eng


def values_at(eng, offset):
    d = (eng["t0"] + timedelta(hours=offset)).date()
    rec = eng["daily"].get(d) or eng["daily"][min(eng["daily"], key=lambda k: abs((k - d).days))]
    return rec["values"], rec["source"], d


def snapshot(offset):
    eng = get_engine()
    vals, src, d = values_at(eng, offset)
    vals, src = np.array(vals, float), list(src)
    t = eng["t0"] + timedelta(hours=offset)
    rt = realtime_now() if offset == 0 else {}
    st = []
    for i, s in enumerate(stations_data):
        pid = ST_IDS[i]
        model_v = float(vals[i])
        r = rt.get(pid)
        if r is not None:                # "ตอนนี้": ใช้ค่าตรวจวัดเรียลไทม์แทนค่าจากโมเดล
            vals[i], src[i] = r["value"], "realtime"
        st.append({"point_id": s["point_id"], "name": s["name"], "district": s["district"], "lat": s["lat"], "lon": s["lon"],
                   "pm25": round(float(vals[i]), 2), "source": src[i], "model_daily": round(model_v, 2),
                   "realtime": ({"value": round(r["value"], 1), "time": r["time"].strftime("%Y-%m-%d %H:%M"), "source": r["source"]} if r else None),
                   "lag_source": eng["info"]["lag_source"][i],
                   "last_obs": LAST_OBS[pid].isoformat() if LAST_OBS.get(pid) else None})
    return {
        "offset": offset, "target_time": t.strftime("%Y-%m-%d %H:00"), "day": d.isoformat(),
        "weather": city_weather(eng["meteo"], offset),
        "provincial_avg": round(float(np.mean(vals)), 2) if N_ST else 0.0,
        "provincial_max": round(float(np.max(vals)), 2) if N_ST else 0.0,
        "provincial_min": round(float(np.min(vals)), 2) if N_ST else 0.0,
        "observed_share": round(sum(s in ("obs", "realtime") for s in src) / max(1, len(src)), 2),
        "realtime_count": sum(s == "realtime" for s in src),
        "stations": st,
    }


def series_payload(eng, o0, o1):
    t0 = eng["t0"]
    offsets = list(range(o0, o1 + 1))
    mats = [values_at(eng, o)[0] for o in offsets]
    trend = [{"offset": o, "iso": (t0 + timedelta(hours=o)).strftime("%Y-%m-%dT%H:00"),
              "time": (t0 + timedelta(hours=o)).strftime("%d/%m %H:00"),
              "pm25": round(float(np.mean(v)), 1) if N_ST else 0.0} for o, v in zip(offsets, mats)]
    M = np.array(mats) if N_ST else np.zeros((len(offsets), 0))
    station_hourly = {s["point_id"]: [round(float(x), 1) for x in M[:, i]] for i, s in enumerate(stations_data)}
    return trend, station_hourly


def meta_payload(eng):
    firms, hsi = eng["firms"], eng["hsi"]
    today = datetime.now(TZ_TH).date()
    daily = [hsi.province_count(today - timedelta(days=k)) for k in range(5)] if hsi.ok else None
    lag_src = eng["info"]["lag_source"]
    return {
        "weather_source": eng["meteo"]["source"], "weather_missing": eng["meteo"].get("missing", []),
        "data_as_of": DATA_AS_OF.isoformat() if DATA_AS_OF else None,
        "rollout_start": eng["info"]["start"].isoformat(),
        "stations_stale": sum(1 for s in lag_src if s != "obs"),
        "station_count": N_ST, "model_loaded": model_data is not None,
        "model_time_step": "daily",
        "hotspot_source": "firms" if firms["status"] == "ok" else firms["status"],
        "hotspot_sensor": hotspot_label(),
        "hotspot_message": firms.get("message", ""),
        "hotspot_summary": {"province_today": daily[0], "province_by_day": daily} if daily else None,
        "grid": {"nx": NX, "ny": NY, "km": GRID_KM, "bounds": GRID_BOUNDS},
        "problems": eng["info"].get("problems", []),
        "realtime": realtime_summary(),
    }


def realtime_summary():
    fresh = realtime_now()
    return {"stations_live": len(fresh), "stations_total": N_ST,
            "last_poll": RT_STATUS["last_poll"].strftime("%Y-%m-%d %H:%M") if RT_STATUS["last_poll"] else None,
            "poll_minutes": REALTIME_POLL_MIN, "log_rows": RT_STATUS["log_rows"], "sources": RT_STATUS["sources"]}


# =============================================================================
# API
# =============================================================================
@app.get("/api/health")
def health():
    """ตรวจสถานะระบบ: เปิด http://localhost:8000/api/health ในเบราว์เซอร์"""
    import sys, sklearn
    eng = get_engine()
    return {"api_version": API_VERSION, "python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__,
            "model_loaded": model_data is not None, "stations": N_ST, "data_as_of": str(DATA_AS_OF),
            "weather_source": eng["meteo"]["source"], "weather_missing": eng["meteo"].get("missing", []),
            "hotspots": eng["firms"]["status"], "hotspot_message": eng["firms"].get("message", ""),
            "rollout_start": str(eng["info"]["start"]), "problems": eng["info"].get("problems", []),
            "realtime": realtime_summary()}


@app.get("/api/realtime")
def get_realtime(refresh: bool = False):
    """ค่า PM2.5 เรียลไทม์ล่าสุดของแต่ละสถานี (?refresh=true เพื่อดึงใหม่ทันที)"""
    if refresh:
        poll_realtime()
        cached_run["timestamp"] = None
        cached_grids.clear()
    fresh = realtime_now()
    rows = [{"point_id": pid, "name": stations_data[i]["name"], "value": round(r["value"], 1),
             "time": r["time"].strftime("%Y-%m-%d %H:%M"), "source": r["source"], "fresh": pid in fresh}
            for i, pid in enumerate(ST_IDS) for r in [RT_LATEST.get(pid)] if r]
    return {"summary": realtime_summary(), "stations": rows}


@app.get("/api/realtime/probe")
def realtime_probe():
    """ตรวจแหล่งข้อมูล JSON: API ตอบอะไรกลับมา ระบบหาช่องข้อมูลเจอหรือไม่ และจับคู่ได้กี่สถานี (ไม่แสดง API key)"""
    out = []
    for src in load_rt_config().get("sources", []):
        if src.get("type") != "json" or not src.get("enabled"):
            continue
        rep = {"name": src.get("name"), "url": src.get("url")}
        try:
            _, has_key = source_headers(src)
            rep["api_key_found"] = has_key
            if src.get("per_station"):
                rep["note"] = "per_station=true: ทดสอบด้วยการดึงจริง"
                rep["matched"] = len(fetch_json_source(src))
            else:
                items, mp, raw = read_json_items(src)
                rep["top_level"] = type(raw).__name__ + (f" keys={list(raw.keys())[:10]}" if isinstance(raw, dict) else f" len={len(raw)}")
                rep["items_found"] = len(items)
                rep["detected_fields"] = mp
                rep["first_item"] = {k: (v if not isinstance(v, (dict, list)) else f"<{type(v).__name__}>")
                                     for k, v in list(items[0].items())[:25]} if items else None
                ids = [str(get_path(it, mp["id"])) for it in items[:2000]] if mp.get("id") else []
                rep["id_examples"] = ids[:8]
                rep["our_ids_found_in_api"] = sorted(set(ids) & set(ST_IDS))[:60]
                rep["matched"] = len(fetch_json_source(src)) if mp.get("value") else 0
        except Exception as e:
            rep["error"] = f"{type(e).__name__}: {e}"[:300]
        out.append(rep)
    return {"sources": out}


@app.get("/api/hotspots/probe")
def hotspots_probe():
    """ตรวจ GISTDA: API ตอบอะไรกลับมา อ่านเวลาจากช่องไหน และนับได้กี่จุดต่อวัน (ไม่แสดง API key)"""
    key = gistda_key()
    if not key:
        return {"provider": "firms", "note": "ไม่พบ gistda_key.txt จึงใช้ NASA FIRMS"}
    rep = {"provider": "gistda", "url": GISTDA_URL}
    try:
        w, s_, e, n = FIRMS_BBOX
        r = requests.get(GISTDA_URL, params={"bbox": f"{w:.6f},{s_:.6f},{e:.6f},{n:.6f}", "limit": 3, "offset": 0, "ct_tn": "ราชอาณาจักรไทย"},
                         headers={"accept": "application/json", "API-Key": key}, timeout=30)
        rep["http_status"] = r.status_code
        j = r.json()
        rep["top_level_keys"] = list(j.keys())[:15] if isinstance(j, dict) else f"list len={len(j)}"
        feats = j.get("features") if isinstance(j, dict) else None
        feats = feats if feats is not None else find_list(j)[1]
        if feats:
            f0 = feats[0]
            rep["first_feature_properties"] = (f0.get("properties", f0) if isinstance(f0, dict) else str(f0))
            rep["first_feature_geometry"] = f0.get("geometry") if isinstance(f0, dict) else None
        pts, info = parse_gistda_features(j)
        rep["detected"] = info
        rep["parsed_example"] = ({**{k: v for k, v in pts[0].items() if k != "time_utc"},
                                  "time_th": pts[0]["time_utc"].astimezone(TZ_TH).strftime("%Y-%m-%d %H:%M")} if pts else None)
        full = fetch_hotspots()
        hsi = HotspotIndex(full)
        today = datetime.now(TZ_TH).date()
        rep["total_points_30d"] = len(full.get("points", []))
        rep["province_by_day_last8"] = {str(today - timedelta(days=k)): hsi.province_count(today - timedelta(days=k)) for k in range(8)}
        rep["status"] = full.get("status"); rep["message"] = full.get("message")
    except Exception as ex:
        rep["error"] = f"{type(ex).__name__}: {ex}"[:300]
    return rep


@app.get("/api/forecast")
def get_forecast():
    eng = get_engine()
    trend, station_hourly = series_payload(eng, 0, FORECAST_HOURS)
    return {"mode": "forecast", "api_version": API_VERSION, "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta_payload(eng), "range": [0, FORECAST_HOURS],
            "hourly_trend": trend, "station_hourly": station_hourly, "backtest_benchmark": BACKTEST}


@app.get("/api/history")
def get_history():
    eng = get_engine()
    trend, station_hourly = series_payload(eng, -HISTORY_HOURS, 0)
    daily = []
    for d in sorted(eng["daily"]):
        if d > datetime.now(TZ_TH).date() or d < (eng["t0"] - timedelta(hours=HISTORY_HOURS)).date():
            continue
        rec = eng["daily"][d]
        daily.append({"date": d.isoformat(), "mean": round(float(np.mean(rec["values"])), 1),
                      "observed_share": round(sum(s == "obs" for s in rec["source"]) / max(1, N_ST), 2),
                      "hotspots_province": eng["hsi"].province_count(d) if eng["hsi"].ok else None})
    return {"mode": "history", "meta": meta_payload(eng), "range": [-HISTORY_HOURS, 0],
            "hourly_trend": trend, "station_hourly": station_hourly, "daily": daily}


cached_grids = {}


@app.get("/api/grid")
def get_grid(offset: int = Query(None, ge=-HISTORY_HOURS, le=FORECAST_HOURS),
             horizon: int = Query(None, ge=0, le=FORECAST_HOURS)):
    """กริด 1x1 กม.: ค่าฝุ่น (IDW จากค่ารายวันของสถานี), ความสูงชั้นผสมและลม (รายชั่วโมง เพื่อแสดงผล)"""
    off = offset if offset is not None else (horizon or 0)
    key = (off, now_hour())
    c = cached_grids.get(key)
    if c and (datetime.now() - c[0]).total_seconds() < 900:
        return c[1]
    eng = get_engine()
    snap = snapshot(off)
    m = eng["meteo"]
    preds = np.array([s["pm25"] for s in snap["stations"]])
    pm = IDW_GRID @ preds if N_ST else np.zeros(NX * NY)
    u, v, spd, _, mh = weather_at(m, off, K_GRID)
    cw = snap["weather"]
    temp_delta = max(0.4, cw["surface_temp"] - cw["temp"] + 1.5)
    mh = mh * (1.0 + min(0.25, 0.04 * temp_delta) * UHI_SHAPE)   # เกาะความร้อนเมือง (แบบจำลอง)
    city_mh = cw["mixing_height"]
    w_up = round(float(math.sqrt(max(0.05, 9.81 * (temp_delta / 300.0) * (city_mh / 120.0)))), 2)
    resp = {
        "offset": off, "time": snap["target_time"], "weather_source": m["source"],
        "grid": {"nx": NX, "ny": NY, "km": GRID_KM, "bounds": GRID_BOUNDS, "order": "lat_major_south_to_north"},
        "pm25": np.round(pm, 1).tolist(), "mixing_height": np.round(mh).astype(int).tolist(),
        "u": np.round(u, 2).tolist(), "v": np.round(v, 2).tolist(),
        "ranges": {"pm25": [round(float(pm.min()), 1), round(float(pm.max()), 1)],
                   "mixing_height": [round(float(mh.min())), round(float(mh.max()))],
                   "wind": [round(float(spd.min()), 2), round(float(spd.max()), 2)]},
        "urban_lift": {"center": [CITY[1], CITY[0]], "updraft_velocity_ms": w_up, "mixing_height": city_mh,
                       "temperature_delta": round(temp_delta, 1),
                       "state": "ยกตัวสูงมาก (ระบายฝุ่นเร็ว)" if w_up >= 1.4 else ("ยกตัวปานกลาง" if w_up >= 0.75 else "อากาศปิดกดทับ (กักฝุ่นใกล้พื้น)")},
        "snapshot": snap,
    }
    if len(cached_grids) > 300:
        cached_grids.clear()
    cached_grids[key] = (datetime.now(), resp)
    return resp


@app.get("/api/hotspots")
def get_hotspots():
    firms = fetch_hotspots()
    out = []
    for i, p in enumerate(firms.get("points", [])):
        district = "นอกพื้นที่สถานี"
        if N_ST:
            d = haversine_km(p["lat"], p["lon"], ST_LAT, ST_LON)
            j = int(np.argmin(d))
            if d[j] <= 20:
                district = stations_data[j]["district"]
        out.append({"id": f"HS-{i + 1:04d}", "lat": round(p["lat"], 5), "lon": round(p["lon"], 5),
                    "frp_mw": round(p["frp_mw"], 1), "confidence": p["confidence"],
                    "satellite": p["satellite"], "source": p["source"], "daynight": p["daynight"],
                    "time_local": p["time_utc"].astimezone(TZ_TH).strftime("%Y-%m-%d %H:%M"),
                    "t": int(p["time_utc"].timestamp()),
                    "in_province": bool(p["in_prov"]) if p.get("in_prov") is not None else bool(in_province(p["lat"], p["lon"])),
                    "district": (f"อ.{p['district_th']}" if p.get("district_th") and not str(p["district_th"]).startswith("อ") else p.get("district_th")) or district})
    return {"status": firms["status"], "message": firms.get("message", ""), "sensor": hotspot_label(),
            "fetched_at": firms.get("fetched_at"), "total_count": len(out), "hotspots": out}


BACKTEST = {
    "overall_r2": 0.8195, "overall_rmse": 10.106, "test_samples": 8486,
    "tier_accuracy": {
        "tier_1": {"f1_pct": 94.1, "recall_pct": 94.1, "precision_pct": 94.2},
        "tier_2": {"f1_pct": 49.3, "recall_pct": 45.3, "precision_pct": 49.3},
        "tier_3": {"f1_pct": 62.1, "recall_pct": 62.1, "precision_pct": 58.6},
        "tier_4": {"f1_pct": 73.2, "recall_pct": 73.2, "precision_pct": 66.6},
        "tier_5": {"f1_pct": 75.4, "recall_pct": 57.6, "precision_pct": 75.4},
    },
    "horizon_confidence_is_estimate": True,
    "horizon_confidence": {
        "plus_0h": {"accuracy_pct": 94.5, "rmse": 4.2}, "plus_6h": {"accuracy_pct": 91.2, "rmse": 6.8},
        "plus_12h": {"accuracy_pct": 87.4, "rmse": 8.3}, "plus_24h": {"accuracy_pct": 82.2, "rmse": 10.1},
        "plus_48h": {"accuracy_pct": 76.5, "rmse": 12.4}, "plus_72h": {"accuracy_pct": 71.8, "rmse": 14.2},
    },
}

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
    for name in ("app.js", "style.css"):
        p = os.path.join(STATIC_DIR, name)
        if os.path.exists(p):
            html = html.replace(f"/static/{name}", f"/static/{name}?v={int(os.path.getmtime(p))}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    import socket
    # ตรวจว่าพอร์ต 8000 ว่าง (ถ้ามีเซิร์ฟเวอร์ตัวเก่ารันค้างอยู่ เบราว์เซอร์จะคุยกับตัวเก่าแทน)
    PORT = int(os.environ.get("PORT", "8000"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        if sk.connect_ex(("127.0.0.1", PORT)) == 0:
            print(f"\n[!] พอร์ต {PORT} มีโปรแกรมอื่นใช้อยู่ (อาจเป็น app.py ตัวเก่าที่ยังรันค้าง)")
            print("    ปิดหน้าต่าง Terminal เดิม หรือรันคำสั่ง:  lsof -ti:8000 | xargs kill   แล้วเริ่มใหม่\n")
            raise SystemExit(1)
    print("กำลังทดสอบระบบก่อนเปิดเซิร์ฟเวอร์ (ค่าเรียลไทม์ อากาศ จุดความร้อน และโมเดล)...")
    try:
        for st_ in poll_realtime():
            print(f"  เรียลไทม์ {st_['name']}: " + (f"{st_['stations']} สถานี" if st_["ok"] else f"ไม่สำเร็จ — {st_['message']}"))
    except Exception:
        traceback.print_exc()
    threading.Thread(target=lambda: (threading.Event().wait(REALTIME_POLL_MIN * 60), realtime_poller()), daemon=True).start()
    try:
        _e = get_engine()
        print(f"  อากาศ: {_e['meteo']['source']}  |  จุดความร้อน ({hotspot_label()}): {_e['firms']['status']}"
              + (f" {len(_e['firms'].get('points', []))} จุด" if _e['firms']['status'] == 'ok' else f" — {_e['firms'].get('message', '')}")
              + f"  |  เริ่มคำนวณจาก: {_e['info']['start']}")
        for p in _e["info"].get("problems", []):
            print("  [!] " + p)
        snapshot(0)
        print(f"  พร้อมใช้งาน -> เปิด http://localhost:{PORT}")
    except Exception:
        print("  [!] ทดสอบไม่ผ่าน — รายละเอียด:")
        traceback.print_exc()
    uvicorn.run(app, host="0.0.0.0", port=PORT)

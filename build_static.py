"""สร้างเว็บแบบ static สำหรับ GitHub Pages (รันโดย GitHub Actions ทุกชั่วโมง)
- ใช้ app.py ตัวเดิมทั้งหมด (โมเดล, อากาศ, จุดความร้อน, ค่าเรียลไทม์) ไม่แก้ตรรกะ
- เขียนผลลงโฟลเดอร์ site/ : index.html + static/ + data/bundle.json + data/hotspots.json
- หน้าเว็บคำนวณกริด 1 กม. เองจาก bundle.json ด้วยสูตรเดียวกับ /api/grid"""
import os, json, shutil, time, re
from datetime import timedelta
import app as A

OUT = os.path.join(A.BASE_DIR, "site")


def r(x, n=2):
    return [round(float(v), n) for v in x]


def build():
    t_start = time.time()
    print("1) ดึงค่าเรียลไทม์")
    rt_status = A.poll_realtime()
    for s in rt_status:
        print(f"   {s['name']}: " + (f"{s['stations']} สถานี" if s["ok"] else f"ไม่สำเร็จ — {s['message']}"))
    A.cached_run["timestamp"] = None

    print("2) รันโมเดล")
    eng = A.get_engine()
    for p in eng["info"].get("problems", []):
        print("   [!]", p)
    fc, hist, hs = A.get_forecast(), A.get_history(), A.get_hotspots()
    print(f"   อากาศ: {eng['meteo']['source']} | จุดความร้อน ({A.hotspot_label()}): {eng['firms']['status']} {len(hs['hotspots'])} จุด")

    print("3) เตรียมข้อมูลรายชั่วโมงสำหรับหน้าเว็บ")
    m = eng["meteo"]
    weather = []
    for off in range(-A.HISTORY_HOURS, A.FORECAST_HOURS + 1):
        i = A.meteo_idx(m, off)
        u, v = A.uv_from(m["ws"][i, :A.N_WX], m["wd"][i, :A.N_WX])
        weather.append({"u": r(u), "v": r(v), "mh": r(m["mh"][i, :A.N_WX], 1),
                        "city": {k: round(x, 3) for k, x in A.city_weather(m, off).items()}})
    daily = {d.isoformat(): {"values": r(rec["values"], 2), "source": list(rec["source"])} for d, rec in eng["daily"].items()}
    realtime = {pid: {"value": round(x["value"], 1), "time": x["time"].strftime("%Y-%m-%d %H:%M"), "source": x["source"]}
                for pid, x in A.realtime_now().items()}
    stations = [{"point_id": s["point_id"], "name": s["name"], "district": s["district"], "lat": s["lat"], "lon": s["lon"],
                 "lag_source": eng["info"]["lag_source"][i],
                 "last_obs": A.LAST_OBS[A.ST_IDS[i]].isoformat() if A.LAST_OBS.get(A.ST_IDS[i]) else None}
                for i, s in enumerate(A.stations_data)]
    geo = {"g_lats": r(A.G_LATS, 6), "g_lons": r(A.G_LONS, 6), "nx": A.NX, "ny": A.NY, "km": A.GRID_KM,
           "bounds": A.GRID_BOUNDS, "wx_lat": r(A.WX_LAT, 6), "wx_lon": r(A.WX_LON, 6), "cos_lat": A.COS_LAT,
           "kernel_bw": 0.16, "idw_power": 2.0, "city": list(A.CITY), "uhi_sigma": 0.07}
    bundle = {"api_version": A.API_VERSION, "built_at": time.strftime("%Y-%m-%d %H:%M"),
              "t0": eng["t0"].strftime("%Y-%m-%d %H:00"), "range": [-A.HISTORY_HOURS, A.FORECAST_HOURS],
              "forecast": fc, "history": hist, "weather": weather, "daily": daily, "realtime": realtime,
              "stations": stations, "geo": geo, "weather_source": m["source"]}

    print("4) เขียนไฟล์เว็บ")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    shutil.copytree(A.STATIC_DIR, os.path.join(OUT, "static"))
    os.makedirs(os.path.join(OUT, "data"))
    with open(os.path.join(OUT, "data", "bundle.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"), default=str)
    with open(os.path.join(OUT, "data", "hotspots.json"), "w", encoding="utf-8") as f:
        json.dump(hs, f, ensure_ascii=False, separators=(",", ":"), default=str)
    with open(os.path.join(A.STATIC_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()
    ver = int(time.time())
    # GitHub Pages อยู่ใต้ /<repo>/ จึงต้องใช้ path แบบสัมพัทธ์ และบอก app.js ว่าเป็นโหมด static
    html = html.replace('href="/static/style.css"', f'href="static/style.css?v={ver}"')
    html = html.replace('<script src="/static/app.js"></script>',
                        f'<script>window.PLK_STATIC = true;</script>\n  <script src="static/app.js?v={ver}"></script>')
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    kb = os.path.getsize(os.path.join(OUT, "data", "bundle.json")) / 1024
    print(f"   site/data/bundle.json {kb:.0f} KB | เสร็จใน {time.time() - t_start:.0f} วินาที")
    return eng


if __name__ == "__main__":
    build()

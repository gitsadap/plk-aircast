# PLK AirCast — พยากรณ์ฝุ่น PM2.5 จังหวัดพิษณุโลก

เว็บแผนที่พยากรณ์ PM2.5 ล่วงหน้า 3 วัน และสืบค้นย้อนหลัง 7 วัน ความละเอียดกริด 1×1 กม.

## การทำงาน

```
GitHub Actions (ทุกชั่วโมง)
  ├─ ดึงค่าเรียลไทม์ DustBoy (CMU CCDC) + Air4Thai → บันทึกถาวรใน realtime_log.csv
  ├─ ดึงอากาศ Open-Meteo + จุดความร้อน GISTDA VIIRS
  ├─ รันโมเดล (app.py) → data/bundle.json
  └─ เผยแพร่บน GitHub Pages
เบราว์เซอร์
  └─ อ่าน bundle.json แล้วคำนวณกริด 1 กม. (IDW + Gaussian kernel) ด้วยสูตรเดียวกับ app.py
```

- **โมเดล:** Extra Trees รายวัน (R² 0.82 จากการทดสอบย้อนหลัง) เดินหน้าทีละวันจากค่าตรวจวัดล่าสุด
- **ค่าตรวจวัด:** `pm25.csv` + ค่าเรียลไทม์ที่สะสมครบวัน (≥ 12 ชม.) ถูกใช้เป็นค่าตรวจวัดรายวันอัตโนมัติ
- **อุตุนิยมวิทยา:** Open-Meteo เฉลี่ยรายวันก่อนป้อนโมเดล (ตรงกับข้อมูล ERA5 รายวันที่ใช้ฝึก)
- **จุดความร้อน:** GISTDA VIIRS 30 วัน นับตามวันปฏิทินและชื่อจังหวัด

## Secrets ที่ต้องตั้ง (Settings → Secrets and variables → Actions)

| ชื่อ | ค่า |
|---|---|
| `GISTDA_API_KEY` | API key ของ GISTDA |
| `CMUCCDC_API_KEY` | API key ของ open-api.cmuccdc.org |

`setup_github.sh` ตั้งค่าให้อัตโนมัติจากไฟล์ `gistda_key.txt` และ `cmuccdc_key.txt` บนเครื่อง (ไฟล์ทั้งสองไม่ถูกอัปโหลด)

## รันบนเครื่อง (โหมดเซิร์ฟเวอร์)

```bash
source .venv/bin/activate
python app.py          # http://localhost:8000
```

ตรวจสถานะ: `/api/health` · ค่าเรียลไทม์: `/api/realtime` · ตรวจแหล่งข้อมูล: `/api/realtime/probe`, `/api/hotspots/probe`

## ข้อจำกัดที่ควรทราบ

- โมเดลฝึกด้วยจุดความร้อน MODIS แต่ระบบใช้ VIIRS ของ GISTDA ซึ่งนับจุดได้มากกว่า
- ค่าพยากรณ์ล่วงหน้าใช้อากาศจากแบบจำลองพยากรณ์ ความแม่นยำจริงจึงต่ำกว่าผลทดสอบย้อนหลังที่ใช้ ERA5
- เว็บอัปเดตทุกชั่วโมง (GitHub อาจเลื่อนเวลารันไปบ้างในช่วงที่มีงานมาก)
- ช่อง `timestamp` ในข้อมูล GISTDA คลาด 7 ชั่วโมง ระบบใช้ `th_date`/`th_time` แทน

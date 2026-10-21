"""บีบอัด pm25_model.joblib (~142 MB -> ~46 MB) ให้อยู่ในขีดจำกัดของ GitHub
สำรองไฟล์เดิมไว้ที่ pm25_model_original.joblib และตรวจว่าผลทำนายเหมือนเดิมทุกค่า"""
import os, shutil, joblib, numpy as np

SRC, BAK = "pm25_model.joblib", "pm25_model_original.joblib"
if os.path.getsize(SRC) < 80e6:
    print("โมเดลถูกบีบอัดแล้ว:", round(os.path.getsize(SRC) / 1e6, 1), "MB"); raise SystemExit
if not os.path.exists(BAK):
    shutil.copy2(SRC, BAK)
data = joblib.load(BAK)
joblib.dump(data, SRC, compress=3)
check = joblib.load(SRC)
X = np.random.RandomState(0).rand(200, len(data["features"])) * 50
same = np.allclose(data["model"].predict(X), check["model"].predict(X))
print(f"บีบอัดแล้ว: {os.path.getsize(BAK)/1e6:.0f} MB -> {os.path.getsize(SRC)/1e6:.0f} MB | ผลทำนายเหมือนเดิม: {same}")
if not same:
    shutil.copy2(BAK, SRC); raise SystemExit("ผลทำนายไม่ตรง — คืนไฟล์เดิมแล้ว")

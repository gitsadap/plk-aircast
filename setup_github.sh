#!/bin/bash
# PLK AirCast — ขึ้น GitHub + เปิด GitHub Pages + ตั้ง Secrets + สั่งสร้างเว็บครั้งแรก (คำสั่งเดียว)
# วิธีใช้:  cd /Volumes/Work_NU03/pm25_plk && bash setup_github.sh
set -e
cd "$(dirname "$0")"
REPO_NAME=plk-aircast

# ความปลอดภัย: ต้องมี .gitignore (กันไฟล์ API key) ก่อนทำอะไรกับ git
if [ ! -f .gitignore ] || ! grep -q "_key.txt" .gitignore; then
  echo "[!] ไม่พบ .gitignore ที่กันไฟล์ API key — หยุดเพื่อความปลอดภัย"; exit 1
fi

echo "===== 1/6 เตรียมโมเดล (บีบอัดให้ต่ำกว่า 100 MB ตามขีดจำกัดของ GitHub)"
source .venv/bin/activate
python compress_model.py 2>/dev/null

echo "===== 2/6 ทดสอบสร้างเว็บบนเครื่องก่อน (ถ้าพังจะหยุดตรงนี้ ไม่อัปโหลด)"
GISTDA_API_KEY="$(cat gistda_key.txt 2>/dev/null)" CMUCCDC_API_KEY="$(cat cmuccdc_key.txt 2>/dev/null)" python build_static.py

echo "===== 3/6 ล็อกอิน GitHub"
if ! command -v gh >/dev/null 2>&1; then
  command -v brew >/dev/null 2>&1 || { echo "[!] ต้องติดตั้ง Homebrew ก่อน: https://brew.sh"; exit 1; }
  brew install gh
fi
if gh auth status >/dev/null 2>&1; then
  gh auth refresh -h github.com -s workflow     # ต้องมีสิทธิ์ workflow จึงอัปโหลดไฟล์ .github/workflows ได้
else
  gh auth login --hostname github.com --git-protocol https --web -s workflow
fi
gh auth setup-git
GH_USER=$(gh api user -q .login)
REPO="$GH_USER/$REPO_NAME"
echo "   บัญชี: $GH_USER"

echo "===== 4/6 อัปโหลดโค้ดขึ้น $REPO"
[ -d .git ] || git init -q -b main
git add -A
if git ls-files | grep -qE "_key\.txt$"; then echo "[!] พบไฟล์ API key ใน git — หยุดเพื่อความปลอดภัย"; exit 1; fi
git -c user.name="$GH_USER" -c user.email="$GH_USER@users.noreply.github.com" commit -qm "PLK AirCast" || true
if gh repo view "$REPO" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$REPO.git"
  git push -u origin main
else
  gh repo create "$REPO" --public --source . --remote origin --push
fi

echo "===== 5/6 ตั้ง Secrets (อ่านจากไฟล์ key บนเครื่อง — ไฟล์ไม่ถูกอัปโหลด)"
[ -f gistda_key.txt ]  && gh secret set GISTDA_API_KEY  --repo "$REPO" < gistda_key.txt  && echo "   ✓ GISTDA_API_KEY"
[ -f cmuccdc_key.txt ] && gh secret set CMUCCDC_API_KEY --repo "$REPO" < cmuccdc_key.txt && echo "   ✓ CMUCCDC_API_KEY"

echo "===== 6/6 เปิด GitHub Pages และสั่งสร้างเว็บครั้งแรก"
gh api -X POST "repos/$REPO/pages" -f build_type=workflow >/dev/null 2>&1 \
  || gh api -X PUT "repos/$REPO/pages" -f build_type=workflow >/dev/null 2>&1 || true
sleep 3
gh workflow run build.yml --repo "$REPO" || echo "   (ระบบจะรันเองจากการ push แล้ว)"

URL="https://$(echo "$GH_USER" | tr '[:upper:]' '[:lower:]').github.io/$REPO_NAME/"
echo ""
echo "================ เสร็จสมบูรณ์ ================"
echo "เว็บไซต์  : $URL"
echo "GitHub   : https://github.com/$REPO"
echo "สถานะ    : https://github.com/$REPO/actions"
echo "(ครั้งแรกใช้เวลาประมาณ 3–5 นาที หลังจากนั้นอัปเดตอัตโนมัติทุกชั่วโมง)"

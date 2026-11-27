import { useEffect, useRef, useState } from "react";
import { HandLandmarker } from "@mediapipe/tasks-vision";
import { useHandTracker } from "../hooks/useHandTracker.js";
import { AMBANG, kotakRoi, periksa } from "../lib/validation.js";
import { IkonCentang, IkonTangan, Kartu } from "./ui.jsx";

const BIRU = "#2f6bef";
const KUNING = "#f0a20b";

const SYARAT = [
  ["terdeteksi", "Tangan terlihat di kamera"],
  ["jari", "Kelima jari terbuka"],
  ["menghadap", "Telapak menghadap kamera"],
  ["jarak", "Jarak tangan pas"],
  ["diam", "Tangan diam"],
];

// Gambar kerangka tangan (dan kotak ROI kalau semua syarat terpenuhi) di atas video.
function gambarTangan(kanvas, titik, valid, lebar, tinggi) {
  if (!kanvas) return;
  kanvas.width = lebar;
  kanvas.height = tinggi;
  const ctx = kanvas.getContext("2d");
  ctx.clearRect(0, 0, lebar, tinggi);
  if (!titik) return;

  ctx.strokeStyle = valid ? BIRU : KUNING;
  ctx.lineWidth = 3;
  for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(titik[start].x * lebar, titik[start].y * tinggi);
    ctx.lineTo(titik[end].x * lebar, titik[end].y * tinggi);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffffff";
  for (const t of titik) {
    ctx.beginPath();
    ctx.arc(t.x * lebar, t.y * tinggi, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  if (valid) {
    const { sudut } = kotakRoi(titik, lebar, tinggi);
    ctx.beginPath();
    sudut.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.fillStyle = "rgba(47,107,239,0.18)";
    ctx.fill();
    ctx.stroke();
  }
}

// Ambil satu frame video sebagai JPEG (data URL). Foto tidak dicerminkan.
function ambilFrame(video, lebar, tinggi) {
  const kanvas = document.createElement("canvas");
  kanvas.width = lebar;
  kanvas.height = tinggi;
  kanvas.getContext("2d").drawImage(video, 0, 0, lebar, tinggi);
  return kanvas.toDataURL("image/jpeg", 0.92);
}

// Kamera dengan pemeriksaan tangan. Foto diambil otomatis saat semua syarat terpenuhi.
//   mode "sekali"   : ambil satu foto lalu berhenti
//   mode "beruntun" : ambil foto setiap AMBANG.jedaMs selama tangan tetap valid
//   onAmbil(foto, { sisi }) dipanggil setiap kali foto diambil
//   samping : isi tambahan di kartu samping
//   kilat   : naikkan angkanya untuk membuat bingkai kamera berkedip
export default function PalmCamera({
  mode = "sekali",
  onAmbil,
  aktif = true,
  samping,
  kilat = 0,
  pesanTahan = "Tahan posisi ini…",
}) {
  const videoRef = useRef(null);
  const kanvasRef = useRef(null);
  const { tangan, siap, galat: galatDetektor } = useHandTracker(videoRef);
  const [galatKamera, setGalatKamera] = useState(null);
  const [status, setStatus] = useState(() => periksa(null));
  const [sibuk, setSibuk] = useState(false);

  const titikSebelum = useRef(null);
  const mulaiTahan = useRef(null); // kapan tangan mulai memenuhi semua syarat
  const bolehAmbilLagi = useRef(0); // waktu paling cepat untuk foto berikutnya
  const selesai = useRef(false); // mode "sekali": foto sudah diambil
  const sibukRef = useRef(false);
  const onAmbilRef = useRef(onAmbil);
  useEffect(() => {
    onAmbilRef.current = onAmbil;
  }, [onAmbil]);

  // Nyalakan kamera sekali, matikan saat halaman ditinggalkan.
  useEffect(() => {
    let stream = null;
    let batal = false;
    navigator.mediaDevices
      .getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      })
      .then((s) => {
        if (batal) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch(() => {
        if (!batal)
          setGalatKamera("Kamera tidak bisa dibuka. Izinkan akses kamera lalu muat ulang halaman.");
      });
    return () => {
      batal = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Setiap kali kamera diaktifkan lagi oleh halaman induk, boleh mengambil foto lagi.
  useEffect(() => {
    if (aktif) {
      selesai.current = false;
      mulaiTahan.current = null;
    }
  }, [aktif]);

  // Setiap ada hasil deteksi baru: cek syarat, gambar kerangka, dan ambil foto kalau sudah waktunya.
  useEffect(() => {
    const video = videoRef.current;
    const lebar = video?.videoWidth || 640;
    const tinggi = video?.videoHeight || 480;
    const titik = tangan?.titik ?? null;
    const p = periksa(titik, tangan?.label, titikSebelum.current, lebar, tinggi);
    titikSebelum.current = titik;
    setStatus(p);
    gambarTangan(kanvasRef.current, titik, p.semua, lebar, tinggi);

    if (!aktif || !siap || selesai.current || sibukRef.current) return;
    if (!p.semua) {
      mulaiTahan.current = null;
      return;
    }
    const sekarang = performance.now();
    mulaiTahan.current ??= sekarang;
    const cukupLama = sekarang - mulaiTahan.current >= AMBANG.tahanMs;
    if (!cukupLama || sekarang < bolehAmbilLagi.current) return;

    sibukRef.current = true;
    setSibuk(true);
    Promise.resolve(
      onAmbilRef.current?.(ambilFrame(video, lebar, tinggi), { sisi: p.sisi }),
    ).finally(() => {
      sibukRef.current = false;
      setSibuk(false);
      if (mode === "sekali") selesai.current = true;
      bolehAmbilLagi.current = performance.now() + AMBANG.jedaMs;
    });
  }, [tangan, aktif, siap, mode]);

  const galat = galatKamera ?? galatDetektor;
  const memuat = !siap && !galat;
  const siapAmbil = aktif && siap && status.semua;
  let pesan = status.pesan;
  if (memuat) pesan = "Menyiapkan kamera dan detektor tangan…";
  else if (sibuk) pesan = "Memproses foto…";
  else if (siapAmbil) pesan = pesanTahan;
  const syaratGagalPertama = SYARAT.find(([kunci]) => !status[kunci])?.[0];

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_20rem]">
      <div className="relative aspect-video overflow-hidden rounded-[28px] bg-ink shadow-float">
        {/* Video dan kanvas dicerminkan (-scale-x-100) supaya terasa seperti bercermin. */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full -scale-x-100 object-cover"
        />
        <canvas
          ref={kanvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-cover"
        />

        {!galat && !memuat && !status.terdeteksi && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div className="flex aspect-[3/4] h-3/5 items-center justify-center rounded-[40px] border-2 border-dashed border-white/50">
              <IkonTangan className="h-1/2 w-1/2 text-white/40" />
            </div>
          </div>
        )}

        {kilat > 0 && (
          <div
            key={kilat}
            className="animasi-kilat pointer-events-none absolute inset-0 rounded-[28px]"
          />
        )}

        {galat ? (
          <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm font-medium text-white">
            {galat}
          </div>
        ) : (
          <div className="absolute inset-x-0 bottom-4 flex justify-center px-4">
            <div
              aria-live="polite"
              className={`max-w-full truncate rounded-2xl px-4 py-2.5 text-center text-sm font-semibold shadow-lg ${
                siapAmbil || sibuk ? "bg-brand text-white" : "bg-white/95 text-ink"
              }`}
            >
              {pesan}
            </div>
          </div>
        )}
      </div>

      <Kartu className="flex flex-col gap-5 p-5">
        {samping}
        <div>
          <h3 className="mb-3 text-sm font-bold text-ink">Syarat foto</h3>
          <ul className="space-y-1.5 text-sm">
            {SYARAT.map(([kunci, teks]) => {
              const ok = status[kunci];
              const sekarang = !memuat && kunci === syaratGagalPertama;
              return (
                <li
                  key={kunci}
                  className={`flex items-center gap-3 rounded-xl px-2.5 py-1.5 ${
                    sekarang
                      ? "bg-warn-soft font-semibold text-ink"
                      : ok
                        ? "text-ink"
                        : "text-muted"
                  }`}
                >
                  <span
                    className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full ${
                      ok ? "bg-brand text-white" : "border-2 border-line"
                    }`}
                  >
                    {ok && <IkonCentang className="h-3 w-3" />}
                  </span>
                  {teks}
                </li>
              );
            })}
          </ul>
        </div>
        <p className="text-xs leading-relaxed text-muted">
          Foto diambil otomatis saat semua syarat terpenuhi dan tangan ditahan sekitar 1 detik.
        </p>
      </Kartu>
    </div>
  );
}

import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router";
import PalmCamera from "../components/PalmCamera.jsx";
import ProcessingPanel from "../components/ProcessingPanel.jsx";
import { JudulHalaman, Pesan } from "../components/ui.jsx";
import { ambilPengaturan, scan } from "../lib/api.js";

export default function Scan() {
  const navigate = useNavigate();
  const [pengaturan, setPengaturan] = useState(null);
  const [batas, setBatas] = useState(0.5);
  const [memproses, setMemproses] = useState(false);
  const [foto, setFoto] = useState(null);
  const [galat, setGalat] = useState(null);

  useEffect(() => {
    ambilPengaturan()
      .then((p) => {
        setPengaturan(p);
        setBatas(p.batas_bawaan);
      })
      .catch((err) => setGalat(`Server tidak bisa dihubungi: ${err.message}`));
  }, []);

  async function onAmbil(fotoBaru) {
    setFoto(fotoBaru);
    setMemproses(true);
    setGalat(null);
    try {
      const hasil = await scan(fotoBaru, batas);
      navigate("/hasil", { state: { hasil } });
    } catch (err) {
      setGalat(`${err.message}. Tunjukkan telapak sekali lagi untuk mencoba ulang.`);
      setMemproses(false);
    }
  }

  const batasMin = pengaturan?.batas_min ?? 0.05;
  const batasMaks = pengaturan?.batas_maks ?? 1;

  return (
    <div className="space-y-8">
      <JudulHalaman
        judul="Siapa pemilik telapak ini?"
        keterangan="Tunjukkan telapak tangan ke kamera. Foto diambil otomatis, lalu sistem mencocokkannya dengan data terdaftar."
      >
        <label className="w-full max-w-xs rounded-2xl border border-white bg-white/80 p-4 text-sm shadow-card sm:w-72">
          <span className="flex items-baseline justify-between">
            <span className="font-semibold text-ink">Batas kecocokan</span>
            <span className="text-base font-bold text-brand tabular-nums">{batas.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={batasMin}
            max={batasMaks}
            step="0.01"
            value={batas}
            onChange={(e) => setBatas(Number(e.target.value))}
            className="mt-3 w-full cursor-pointer accent-brand"
          />
          <span className="mt-2 flex justify-between text-xs text-muted">
            <span>Lebih ketat</span>
            <span>Lebih longgar</span>
          </span>
        </label>
      </JudulHalaman>

      {pengaturan && !pengaturan.model_siap && (
        <Pesan jenis="peringatan">
          Model belum dibuat. Scan tetap berjalan sampai tahap potong ROI, tetapi belum bisa mengenali orang. Isi{" "}
          <code>backend/model.py</code> untuk memasang model.
        </Pesan>
      )}

      {/* Kamera tetap dipasang (disembunyikan) supaya detektor tangan tidak dimuat ulang. */}
      <div className={memproses ? "hidden" : ""}>
        <PalmCamera mode="sekali" onAmbil={onAmbil} aktif={!memproses} />
      </div>
      {memproses && (
        <ProcessingPanel foto={foto} judul="Mencocokkan telapak…" keterangan="Foto sudah diambil. Tangan boleh diturunkan." />
      )}
      {galat && <Pesan jenis="galat">{galat}</Pesan>}

      <p className="text-sm text-muted">
        Belum terdaftar?{" "}
        <Link to="/daftar" className="font-semibold text-brand hover:underline">
          Daftarkan telapakmu
        </Link>
      </p>
    </div>
  );
}

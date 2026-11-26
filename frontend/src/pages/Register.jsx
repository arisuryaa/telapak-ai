import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import PalmCamera from "../components/PalmCamera.jsx";
import ProcessingPanel from "../components/ProcessingPanel.jsx";
import { IkonCentang, IkonTangan, JudulHalaman, Kartu, Pesan, input, tombol } from "../components/ui.jsx";
import { ambilPengaturan, daftarOrang, kirimSampel, simpanOrang } from "../lib/api.js";
import { formatTangan } from "../lib/format.js";

const JUMLAH_SAMPEL = 8;
const STATUS = ["mahasiswa", "dosen", "staf", "tamu"];
const FORM_AWAL = { nama: "", umur: "", status: "mahasiswa", sisi: "kanan", persetujuan: false, ganti: false };

// Delapan kotak kecil: kotak terisi biru untuk setiap sampel yang sudah diambil.
function ProgresSampel({ jumlah }) {
  return (
    <div className="space-y-2 text-center">
      <p className="text-4xl font-extrabold text-ink tabular-nums">
        {jumlah}
        <span className="text-lg font-bold text-muted">/{JUMLAH_SAMPEL}</span>
      </p>
      <div className="flex gap-1.5">
        {Array.from({ length: JUMLAH_SAMPEL }, (_, i) => (
          <span key={i} className={`h-2 flex-1 rounded-full ${i < jumlah ? "bg-brand" : "bg-lilac"}`} />
        ))}
      </div>
      <p className="text-xs text-muted">sampel diambil</p>
    </div>
  );
}

function PilihanTangan({ nilai, onUbah, terdaftar }) {
  return (
    <fieldset>
      <legend className="mb-2 text-sm font-semibold text-ink">Tangan yang didaftarkan</legend>
      <div className="grid grid-cols-2 gap-3">
        {["kiri", "kanan"].map((sisi) => (
          <label
            key={sisi}
            className={`relative flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 px-3 py-4 text-sm font-semibold ${
              nilai === sisi ? "border-brand bg-brand-soft text-brand" : "border-line bg-white text-muted"
            }`}
          >
            <input
              type="radio"
              name="sisi"
              checked={nilai === sisi}
              onChange={() => onUbah(sisi)}
              className="sr-only"
            />
            <IkonTangan cermin={sisi === "kiri"} className="h-9 w-9" />
            Tangan {sisi}
            {terdaftar.includes(sisi) && (
              <span className="absolute top-2 right-2 rounded-full bg-warn-soft px-2 py-0.5 text-[11px] text-warn">
                terdaftar
              </span>
            )}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export default function Register() {
  const [pengaturan, setPengaturan] = useState(null);
  const [daftar, setDaftar] = useState([]);
  const [pilihan, setPilihan] = useState("baru"); // "baru" atau id orang yang sudah terdaftar
  const [form, setForm] = useState(FORM_AWAL);
  const [tahap, setTahap] = useState("form"); // "form" -> "ambil" -> "selesai"
  const [jumlah, setJumlah] = useState(0);
  const [ditolak, setDitolak] = useState(null);
  const [galat, setGalat] = useState(null);
  const [menyimpan, setMenyimpan] = useState(false);
  const [tersimpan, setTersimpan] = useState(null);
  const sampel = useRef([]); // embedding dari setiap foto yang diterima

  useEffect(() => {
    ambilPengaturan()
      .then(setPengaturan)
      .catch((err) => setGalat(err.message));
    daftarOrang()
      .then(setDaftar)
      .catch((err) => setGalat(err.message));
  }, []);

  const orangLama = daftar.find((o) => String(o.id) === pilihan);
  const tanganTerdaftar = orangLama ? orangLama.tangan.map((t) => t.sisi) : [];
  const sudahTerdaftar = tanganTerdaftar.includes(form.sisi);
  const modelBelumAda = pengaturan && !pengaturan.model_siap;

  function ubah(kunci, nilai) {
    setForm((f) => ({ ...f, [kunci]: nilai }));
  }

  function ulangiSampel() {
    sampel.current = [];
    setJumlah(0);
    setDitolak(null);
    setGalat(null);
  }

  function mulai(e) {
    e.preventDefault();
    setGalat(null);
    if (pilihan === "baru") {
      const umur = Number(form.umur);
      if (!form.nama.trim()) return setGalat("Isi nama dulu.");
      if (!Number.isInteger(umur) || umur < 1 || umur > 120) return setGalat("Umur harus angka 1–120.");
    }
    if (sudahTerdaftar && !form.ganti) {
      return setGalat(`Centang "Ganti data lama" untuk mendaftarkan ulang tangan ${form.sisi}.`);
    }
    if (!form.persetujuan) return setGalat("Centang persetujuan penyimpanan data dulu.");
    ulangiSampel();
    setTahap("ambil");
  }

  // Dipanggil kamera setiap kali satu foto diambil.
  async function onAmbil(foto) {
    try {
      const hasil = await kirimSampel(foto);
      if (!hasil.valid) return setDitolak(hasil.pesan);
      if (hasil.sisi !== form.sisi) return setDitolak(`Itu tangan ${hasil.sisi}. Tunjukkan tangan ${form.sisi}.`);
      setDitolak(null);
      sampel.current.push(hasil.embedding);
      setJumlah(sampel.current.length);
      if (sampel.current.length === JUMLAH_SAMPEL) await simpan();
    } catch (err) {
      setDitolak(err.message);
    }
  }

  async function simpan() {
    setMenyimpan(true);
    try {
      const hasil = await simpanOrang({
        orang_id: orangLama?.id,
        nama: form.nama.trim(),
        umur: Number(form.umur),
        status: form.status,
        sisi: form.sisi,
        persetujuan: true,
        ganti: form.ganti,
        embedding: sampel.current,
      });
      setTersimpan(hasil.orang);
      setTahap("selesai");
    } catch (err) {
      setGalat(`Gagal menyimpan: ${err.message}`);
    } finally {
      setMenyimpan(false);
    }
  }

  if (tahap === "selesai") {
    return (
      <Kartu className="mx-auto max-w-xl p-8 text-center">
        <span className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-ok-soft text-ok">
          <IkonCentang className="h-8 w-8" />
        </span>
        <h1 className="mt-5 text-3xl font-extrabold tracking-tight text-ink">Telapak terdaftar</h1>
        <p className="mt-2 text-muted">
          Tangan {form.sisi} milik <b className="text-ink">{tersimpan.nama}</b> tersimpan dengan {JUMLAH_SAMPEL}{" "}
          sampel. Sekarang coba scan untuk memastikan sistem mengenalimu.
        </p>
        <Link to="/" className={`${tombol.utama} mt-7`}>
          Coba scan sekarang
        </Link>
      </Kartu>
    );
  }

  if (tahap === "ambil") {
    const nama = orangLama?.nama ?? form.nama.trim();
    return (
      <div className="space-y-6">
        <JudulHalaman
          judul={`Tunjukkan telapak tangan ${form.sisi}`}
          keterangan={`Untuk ${nama}. Tahan tangan diam di depan kamera sampai ${JUMLAH_SAMPEL} sampel terkumpul.`}
        >
          <button type="button" onClick={() => setTahap("form")} disabled={menyimpan} className={tombol.kedua}>
            Batal
          </button>
        </JudulHalaman>

        <div className={menyimpan ? "hidden" : ""}>
          <PalmCamera
            mode="beruntun"
            onAmbil={onAmbil}
            aktif={!menyimpan && !galat && jumlah < JUMLAH_SAMPEL}
            kilat={jumlah}
            pesanTahan={`Tahan… mengambil sampel ${Math.min(jumlah + 1, JUMLAH_SAMPEL)} dari ${JUMLAH_SAMPEL}`}
            samping={
              <div className="space-y-4">
                <ProgresSampel jumlah={jumlah} />
                {ditolak && (
                  <Pesan jenis="peringatan" className="text-xs">
                    Sampel tadi tidak dipakai: {ditolak}
                  </Pesan>
                )}
              </div>
            }
          />
        </div>
        {menyimpan && <ProcessingPanel judul="Menyimpan data…" keterangan="Tangan boleh diturunkan." />}
        {galat && (
          <Pesan jenis="galat" className="flex flex-wrap items-center justify-between gap-3">
            <span>{galat}</span>
            <button type="button" onClick={ulangiSampel} className={tombol.utama}>
              Ulangi pengambilan
            </button>
          </Pesan>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <JudulHalaman
        judul="Daftarkan telapak tangan"
        keterangan={`Isi data diri, lalu tunjukkan telapak ke kamera. Sistem mengambil ${JUMLAH_SAMPEL} foto otomatis.`}
      />

      {modelBelumAda && (
        <Pesan jenis="peringatan">
          Model belum dibuat, jadi belum bisa mendaftar. Isi <code>backend/model.py</code> lalu jalankan ulang
          backend.
        </Pesan>
      )}

      <Kartu className="max-w-2xl p-6 sm:p-8">
        <form onSubmit={mulai} className="space-y-6" noValidate>
          <label className="block space-y-2 text-sm">
            <span className="font-semibold text-ink">Siapa yang didaftarkan?</span>
            <select
              value={pilihan}
              onChange={(e) => {
                setPilihan(e.target.value);
                ubah("ganti", false);
              }}
              className={input}
            >
              <option value="baru">Orang baru</option>
              {daftar.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.nama} (sudah terdaftar: {formatTangan(o) || "-"})
                </option>
              ))}
            </select>
          </label>

          {pilihan === "baru" && (
            <div className="space-y-4">
              <label className="block space-y-2 text-sm">
                <span className="font-semibold text-ink">Nama lengkap</span>
                <input
                  value={form.nama}
                  onChange={(e) => ubah("nama", e.target.value)}
                  maxLength={100}
                  placeholder="Contoh: Ayu Lestari"
                  className={input}
                />
              </label>
              <div className="grid grid-cols-2 gap-4">
                <label className="block space-y-2 text-sm">
                  <span className="font-semibold text-ink">Umur</span>
                  <input
                    type="number"
                    min="1"
                    max="120"
                    value={form.umur}
                    onChange={(e) => ubah("umur", e.target.value)}
                    placeholder="21"
                    className={input}
                  />
                </label>
                <label className="block space-y-2 text-sm">
                  <span className="font-semibold text-ink">Status</span>
                  <select value={form.status} onChange={(e) => ubah("status", e.target.value)} className={input}>
                    {STATUS.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
          )}

          <PilihanTangan
            nilai={form.sisi}
            onUbah={(sisi) => {
              ubah("sisi", sisi);
              ubah("ganti", false);
            }}
            terdaftar={tanganTerdaftar}
          />

          {sudahTerdaftar && (
            <label className="flex items-start gap-3 rounded-2xl bg-warn-soft p-4 text-sm text-ink">
              <input
                type="checkbox"
                checked={form.ganti}
                onChange={(e) => ubah("ganti", e.target.checked)}
                className="mt-0.5 h-4 w-4 accent-brand"
              />
              <span>
                <b>Ganti data lama.</b> Tangan {form.sisi} orang ini sudah terdaftar dan datanya akan diganti.
              </span>
            </label>
          )}

          <label className="flex items-start gap-3 text-sm text-muted">
            <input
              type="checkbox"
              checked={form.persetujuan}
              onChange={(e) => ubah("persetujuan", e.target.checked)}
              className="mt-0.5 h-4 w-4 accent-brand"
            />
            <span>Saya setuju embedding telapak saya disimpan untuk keperluan demo. Foto tidak disimpan.</span>
          </label>

          {galat && <Pesan jenis="galat">{galat}</Pesan>}

          <button type="submit" disabled={modelBelumAda} className={`${tombol.utama} w-full py-3.5 sm:w-auto`}>
            Nyalakan kamera dan mulai
          </button>
        </form>
      </Kartu>
    </div>
  );
}

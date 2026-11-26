import { useEffect, useState } from "react";
import { Link } from "react-router";
import { JudulHalaman, Kartu, Pesan, tombol } from "../components/ui.jsx";
import { daftarOrang, hapusOrang, hapusSemua } from "../lib/api.js";
import { formatTangan } from "../lib/format.js";

const formatTanggal = (iso) =>
  new Date(iso).toLocaleDateString("id-ID", { day: "numeric", month: "long", year: "numeric" });

export default function RegisteredData() {
  const [daftar, setDaftar] = useState(null);
  const [galat, setGalat] = useState(null);

  async function muat() {
    try {
      setDaftar(await daftarOrang());
      setGalat(null);
    } catch (err) {
      setGalat(err.message);
    }
  }

  useEffect(() => {
    muat();
  }, []);

  async function hapus(orang) {
    if (!window.confirm(`Hapus data ${orang.nama}?`)) return;
    await hapusOrang(orang.id).catch((err) => setGalat(err.message));
    muat();
  }

  async function hapusSemuanya() {
    if (!window.confirm("Hapus SEMUA data terdaftar?")) return;
    await hapusSemua().catch((err) => setGalat(err.message));
    muat();
  }

  return (
    <div className="space-y-8">
      <JudulHalaman judul="Data terdaftar" keterangan={daftar && `${daftar.length} orang terdaftar.`}>
        <div className="flex flex-wrap gap-3">
          {daftar?.length > 0 && (
            <button type="button" onClick={hapusSemuanya} className={tombol.kedua}>
              <span className="text-err">Hapus semua</span>
            </button>
          )}
          <Link to="/daftar" className={tombol.utama}>
            Daftarkan orang baru
          </Link>
        </div>
      </JudulHalaman>

      {galat && <Pesan jenis="galat">{galat}</Pesan>}
      {!daftar && !galat && <p className="text-muted">Memuat…</p>}
      {daftar?.length === 0 && (
        <Kartu className="px-6 py-14 text-center text-muted">Belum ada telapak terdaftar.</Kartu>
      )}

      {daftar?.length > 0 && (
        <Kartu className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted">
              <tr>
                <th className="px-6 pt-5 pb-3">Nama</th>
                <th className="px-4 pt-5 pb-3">Umur</th>
                <th className="px-4 pt-5 pb-3">Status</th>
                <th className="px-4 pt-5 pb-3">Tangan (model)</th>
                <th className="px-4 pt-5 pb-3">Tanggal daftar</th>
                <th className="px-6 pt-5 pb-3" />
              </tr>
            </thead>
            <tbody>
              {daftar.map((o) => (
                <tr key={o.id} className="border-t border-line">
                  <td className="px-6 py-3.5">
                    <Link to={`/orang/${o.id}`} className="font-semibold text-ink hover:text-brand">
                      {o.nama}
                    </Link>
                  </td>
                  <td className="px-4 py-3.5 tabular-nums">{o.umur}</td>
                  <td className="px-4 py-3.5 capitalize">{o.status}</td>
                  <td className="px-4 py-3.5 text-muted">{formatTangan(o) || "-"}</td>
                  <td className="px-4 py-3.5 text-muted">{formatTanggal(o.dibuat_pada)}</td>
                  <td className="px-6 py-3.5 text-right">
                    <button
                      type="button"
                      onClick={() => hapus(o)}
                      className="rounded-full px-3 py-1.5 font-semibold text-err hover:bg-err-soft"
                    >
                      Hapus
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Kartu>
      )}
    </div>
  );
}

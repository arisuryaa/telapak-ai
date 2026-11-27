import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router";
import { Kartu, Pesan, tombol } from "../components/ui.jsx";
import { ambilOrang } from "../lib/api.js";
import { formatTangan } from "../lib/format.js";

export default function PersonDetail() {
  const { id } = useParams();
  const { state } = useLocation(); // berisi hasil scan kalau datang dari halaman Hasil
  const [orang, setOrang] = useState(null);
  const [galat, setGalat] = useState(null);

  useEffect(() => {
    ambilOrang(id)
      .then(setOrang)
      .catch((err) => setGalat(err.status === 404 ? "Orang ini tidak ditemukan." : err.message));
  }, [id]);

  if (galat) return <Pesan jenis="galat">{galat}</Pesan>;
  if (!orang) return <p className="text-muted">Memuat…</p>;

  const rincian = [
    ["Umur", `${orang.umur} tahun`],
    ["Status", orang.status],
    ["Tangan terdaftar", formatTangan(orang) || "-"],
  ];

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <Kartu className="overflow-hidden">
        <h1 className="p-6 text-3xl font-extrabold tracking-tight text-ink sm:p-8">{orang.nama}</h1>
        <dl className="grid border-t border-line sm:grid-cols-3">
          {rincian.map(([nama, nilai]) => (
            <div key={nama} className="px-6 py-4 sm:px-8">
              <dt className="text-xs font-semibold text-muted">{nama}</dt>
              <dd className="mt-1 font-semibold text-ink capitalize">{nilai}</dd>
            </div>
          ))}
        </dl>
        {state?.terdekat && (
          <p className="border-t border-line bg-ok-soft px-6 py-4 text-sm font-medium text-ok sm:px-8">
            Cocok dengan tangan {state.terdekat.sisi} pada scan terakhir, jarak{" "}
            {state.terdekat.jarak.toFixed(3)}.
          </p>
        )}
      </Kartu>
      <Link to="/" className={tombol.utama}>
        Scan lagi
      </Link>
    </div>
  );
}

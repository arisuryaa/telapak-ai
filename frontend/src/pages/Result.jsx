import { Link, Navigate, useLocation } from "react-router";
import DistanceChart from "../components/DistanceChart.jsx";
import { IkonCentang, IkonSilang, Kartu, tombol } from "../components/ui.jsx";

const WARNA_STATUS = {
  dikenali: { ikon: "bg-ok text-white", latar: "bg-ok-soft" },
  tidak_dikenal: { ikon: "bg-warn-bar text-white", latar: "bg-warn-soft" },
  gagal: { ikon: "bg-err text-white", latar: "bg-err-soft" },
};

// Satu kartu langkah pemrosesan: judul, penjelasan, gambar, dan angka-angkanya.
function Langkah({ langkah, terakhir }) {
  return (
    <li className="relative grid grid-cols-[2.5rem_1fr] gap-4">
      {!terakhir && (
        <span className="absolute top-11 -bottom-5 left-5 w-0.5 -translate-x-1/2 bg-line" aria-hidden="true" />
      )}
      <span className="relative z-10 flex h-10 w-10 items-center justify-center rounded-full bg-brand text-sm font-bold text-white shadow-float">
        {langkah.no}
      </span>
      <Kartu className="min-w-0 p-5">
        <h2 className="text-lg font-bold tracking-tight text-ink">{langkah.judul}</h2>
        <p className="mt-1 mb-4 text-sm text-muted">{langkah.penjelasan}</p>

        {langkah.jarak && <DistanceChart jarak={langkah.jarak} batas={langkah.angka.batas} />}

        {langkah.gambar.length > 0 && (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {langkah.gambar.map((g) => (
              <figure key={g.label} className="text-center text-xs text-muted">
                <img src={g.src} alt={g.label} className="max-h-80 w-full rounded-xl bg-lilac object-contain" />
                <figcaption className="mt-1.5 font-medium">{g.label}</figcaption>
              </figure>
            ))}
          </div>
        )}

        <dl className="mt-4 flex flex-wrap gap-2 text-xs">
          {Object.entries(langkah.angka).map(([nama, nilai]) => (
            <div key={nama} className="flex gap-1.5 rounded-full bg-canvas px-3 py-1">
              <dt className="text-muted">{nama}</dt>
              <dd className="font-semibold text-ink tabular-nums">{String(nilai)}</dd>
            </div>
          ))}
        </dl>
      </Kartu>
    </li>
  );
}

function KotakHasil({ hasil }) {
  const { status, pesan, terdekat, batas } = hasil;
  const dikenali = status === "dikenali";
  const warna = WARNA_STATUS[status];

  let keterangan = "Belum ada data terdaftar untuk dibandingkan.";
  if (status === "gagal") keterangan = "Foto berhenti diproses di langkah terakhir di bawah.";
  else if (dikenali) keterangan = `Tangan ${terdekat.sisi}, jarak ${terdekat.jarak.toFixed(3)} (batas ${batas.toFixed(2)})`;
  else if (terdekat) {
    keterangan = `Paling mirip: ${terdekat.nama} (tangan ${terdekat.sisi}), jarak ${terdekat.jarak.toFixed(3)}`;
  }

  return (
    <Kartu className="overflow-hidden">
      <div className={`flex flex-wrap items-center gap-5 p-6 sm:p-8 ${warna.latar}`}>
        <span className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-full ${warna.ikon}`}>
          {dikenali ? <IkonCentang className="h-7 w-7" /> : <IkonSilang className="h-7 w-7" />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-2xl font-extrabold tracking-tight text-ink sm:text-3xl">{pesan}</p>
          <p className="mt-1 text-sm text-muted">{keterangan}</p>
        </div>
        <div className="flex flex-wrap gap-3">
          {dikenali && (
            <Link to={`/orang/${terdekat.orang_id}`} state={{ terdekat }} className={tombol.utama}>
              Lihat detail
            </Link>
          )}
          <Link to="/" className={dikenali ? tombol.kedua : tombol.utama}>
            Scan ulang
          </Link>
        </div>
      </div>
    </Kartu>
  );
}

export default function Result() {
  const { state } = useLocation();
  if (!state?.hasil) return <Navigate to="/" replace />;
  const hasil = state.hasil;

  return (
    <div className="space-y-10">
      <KotakHasil hasil={hasil} />
      <section className="space-y-5">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-ink">Cara sistem sampai ke jawaban ini</h2>
          <p className="mt-1 text-muted">Gambar di bawah adalah data asli dari setiap tahap pemrosesan foto tadi.</p>
        </div>
        <ol className="space-y-5">
          {hasil.langkah.map((l, i) => (
            <Langkah key={l.no} langkah={l} terakhir={i === hasil.langkah.length - 1} />
          ))}
        </ol>
      </section>
    </div>
  );
}

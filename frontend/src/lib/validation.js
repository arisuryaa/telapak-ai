// Aturan tangan yang dicek di browser, sama seperti backend/validation.py.
// Titik tangan dari MediaPipe bernilai 0..1; lebar dan tinggi adalah ukuran video dalam piksel.

export const AMBANG = {
  rasioJari: 1.1, // ujung jari harus lebih jauh dari pergelangan daripada ruas tengahnya
  rasioJempol: 0.7, // jempol harus cukup jauh dari pangkal telunjuk
  lebarMin: 0.12, // lebar telapak minimal, dibanding lebar video
  lebarMaks: 0.45, // lebar telapak maksimal, dibanding lebar video
  gerakMaks: 0.012, // rata-rata pergeseran titik antar frame supaya dianggap diam
  tahanMs: 1000, // lama tangan harus ditahan sebelum foto diambil
  jedaMs: 800, // jeda antar foto saat mengambil beberapa sampel
};

// (ujung jari, ruas tengah) untuk telunjuk, tengah, manis, kelingking
const PASANGAN_JARI = [
  [8, 6],
  [12, 10],
  [16, 14],
  [20, 18],
];

const piksel = (titik, i, lebar, tinggi) => ({ x: titik[i].x * lebar, y: titik[i].y * tinggi });
const jarak = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// MediaPipe di browser melaporkan tangan sungguhan (tanpa anggapan foto dicerminkan),
// jadi "Right" = kanan. Di backend Python kebalikannya, lihat backend/validation.py.
export function sisiDariLabel(label) {
  return label === "Right" ? "kanan" : "kiri";
}

export function jariTerbuka(titik, lebar, tinggi) {
  const pergelangan = piksel(titik, 0, lebar, tinggi);
  for (const [ujung, ruas] of PASANGAN_JARI) {
    const jarakUjung = jarak(piksel(titik, ujung, lebar, tinggi), pergelangan);
    const jarakRuas = jarak(piksel(titik, ruas, lebar, tinggi), pergelangan);
    if (jarakUjung <= AMBANG.rasioJari * jarakRuas) return false;
  }
  const jempol = jarak(piksel(titik, 4, lebar, tinggi), piksel(titik, 5, lebar, tinggi));
  const lebarTelapak = jarak(piksel(titik, 5, lebar, tinggi), piksel(titik, 17, lebar, tinggi));
  return jempol > AMBANG.rasioJempol * lebarTelapak;
}

// Perkalian silang vektor pergelangan->telunjuk dan pergelangan->kelingking.
// Tandanya membedakan telapak dan punggung tangan, dan terbalik untuk tangan kiri.
export function menghadapKamera(titik, sisi, lebar, tinggi) {
  const p0 = piksel(titik, 0, lebar, tinggi);
  const a = piksel(titik, 5, lebar, tinggi);
  const b = piksel(titik, 17, lebar, tinggi);
  const silang = (a.x - p0.x) * (b.y - p0.y) - (a.y - p0.y) * (b.x - p0.x);
  return sisi === "kanan" ? silang < 0 : silang > 0;
}

// Empat sudut kotak ROI di video, sama seperti palm.potong_roi di backend:
// persegi selebar telapak, dari 0,15 sampai 1,15 lebar telapak di bawah garis pangkal jari.
export function kotakRoi(titik, lebar, tinggi) {
  const p5 = piksel(titik, 5, lebar, tinggi);
  const p17 = piksel(titik, 17, lebar, tinggi);
  const p0 = piksel(titik, 0, lebar, tinggi);
  const pusat = { x: (p5.x + p17.x) / 2, y: (p5.y + p17.y) / 2 };
  const s = jarak(p5, p17);
  const mendatar = { x: (p17.x - p5.x) / s, y: (p17.y - p5.y) / s };
  // arah tegak lurus garis pangkal jari, menuju pergelangan
  let turun = { x: -mendatar.y, y: mendatar.x };
  if ((p0.x - pusat.x) * turun.x + (p0.y - pusat.y) * turun.y < 0) {
    turun = { x: -turun.x, y: -turun.y };
  }

  const sudut = [
    [-0.5, 0.15],
    [0.5, 0.15],
    [0.5, 1.15],
    [-0.5, 1.15],
  ].map(([geser, bawah]) => [
    pusat.x + mendatar.x * geser * s + turun.x * bawah * s,
    pusat.y + mendatar.y * geser * s + turun.y * bawah * s,
  ]);
  const didalam = sudut.every(([x, y]) => x >= 0 && x <= lebar && y >= 0 && y <= tinggi);
  return { sudut, didalam };
}

// Rata-rata pergeseran 21 titik dari frame sebelumnya.
function rataGerak(titikA, titikB) {
  if (!titikA || !titikB) return Infinity;
  let total = 0;
  for (let i = 0; i < titikA.length; i++) {
    total += Math.hypot(titikA[i].x - titikB[i].x, titikA[i].y - titikB[i].y);
  }
  return total / titikA.length;
}

// Cek semua syarat foto. Kembalikan hasil tiap syarat dan pesan untuk pengguna.
export function periksa(titik, label, titikSebelum, lebar, tinggi) {
  if (!titik) {
    return {
      terdeteksi: false,
      jari: false,
      menghadap: false,
      jarak: false,
      diam: false,
      semua: false,
      sisi: null,
      pesan: "Letakkan telapak di depan kamera",
    };
  }
  const sisi = sisiDariLabel(label);
  const jari = jariTerbuka(titik, lebar, tinggi);
  const menghadap = menghadapKamera(titik, sisi, lebar, tinggi);
  const lebarTelapak =
    jarak(piksel(titik, 5, lebar, tinggi), piksel(titik, 17, lebar, tinggi)) / lebar;
  const jarakPas =
    lebarTelapak >= AMBANG.lebarMin &&
    lebarTelapak <= AMBANG.lebarMaks &&
    kotakRoi(titik, lebar, tinggi).didalam;
  const diam = rataGerak(titik, titikSebelum) <= AMBANG.gerakMaks;

  let pesan = "Tahan...";
  if (!jari) pesan = "Buka semua jari";
  else if (!menghadap) pesan = "Tunjukkan telapak tangan, bukan punggung tangan";
  else if (!jarakPas) pesan = "Atur jarak tangan ke kamera";
  else if (!diam) pesan = "Tahan tangan agar diam";

  return {
    terdeteksi: true,
    jari,
    menghadap,
    jarak: jarakPas,
    diam,
    semua: jari && menghadap && jarakPas && diam,
    sisi,
    pesan,
  };
}

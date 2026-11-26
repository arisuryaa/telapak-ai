// Fungsi untuk memanggil API Flask. Dokumentasi API: http://127.0.0.1:5000/api/docs

async function kirim(alamat, opsi = {}) {
  const res = await fetch(alamat, {
    headers: { "Content-Type": "application/json" },
    ...opsi,
  });
  // 204 (berhasil hapus) tidak punya isi JSON.
  const data = res.status === 204 ? null : await res.json();
  if (!res.ok) {
    const err = new Error(data?.pesan ?? `Permintaan gagal (${res.status})`);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

const kirimJson = (alamat, isi) => kirim(alamat, { method: "POST", body: JSON.stringify(isi) });

export const ambilPengaturan = () => kirim("/api/pengaturan");
export const scan = (foto, batas) => kirimJson("/api/scan", { foto, batas });
export const kirimSampel = (foto) => kirimJson("/api/sampel", { foto });
export const simpanOrang = (data) => kirimJson("/api/orang", data);
export const daftarOrang = () => kirim("/api/orang");
export const ambilOrang = (id) => kirim(`/api/orang/${id}`);
export const hapusOrang = (id) => kirim(`/api/orang/${id}`, { method: "DELETE" });
export const hapusSemua = () => kirim("/api/orang", { method: "DELETE" });

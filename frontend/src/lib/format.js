// "kanan (resnet18), kiri (resnet18)": setiap tangan terdaftar beserta model yang membuatnya.
export function formatTangan(orang) {
  return orang.tangan.map((t) => `${t.sisi} (${t.model})`).join(", ");
}

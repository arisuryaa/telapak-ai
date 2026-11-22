// "kanan (model 20261123-0130), kiri (model 20261123-0130)": each enrolled hand with the model version that made it.
export function formatTangan(orang) {
  return orang.tangan.map((sisi) => `${sisi} (model ${orang.versi_tangan?.[sisi] || "?"})`).join(", ");
}

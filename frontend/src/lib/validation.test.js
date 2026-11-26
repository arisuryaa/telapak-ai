import { describe, expect, test } from "vitest";
import contoh from "../../../backend/tests/fixtures/landmarks_test.json";
import { jariTerbuka, kotakRoi, menghadapKamera, periksa, sisiDariLabel } from "./validation.js";

const LEBAR = contoh.lebar;
const TINGGI = contoh.tinggi;

function titik(nama) {
  return contoh[nama].map(([x, y]) => ({ x: x / LEBAR, y: y / TINGGI, z: 0 }));
}

describe("aturan tangan", () => {
  test("label MediaPipe menjadi sisi tangan", () => {
    expect(sisiDariLabel("Right")).toBe("kanan");
    expect(sisiDariLabel("Left")).toBe("kiri");
  });

  test("jari terbuka lolos, mengepal ditolak", () => {
    expect(jariTerbuka(titik("terbuka_kanan"), LEBAR, TINGGI)).toBe(true);
    expect(jariTerbuka(titik("kepal_kanan"), LEBAR, TINGGI)).toBe(false);
  });

  test("telapak lolos, punggung tangan ditolak", () => {
    expect(menghadapKamera(titik("terbuka_kanan"), "kanan", LEBAR, TINGGI)).toBe(true);
    expect(menghadapKamera(titik("punggung_kanan"), "kanan", LEBAR, TINGGI)).toBe(false);
  });

  test("tangan terbuka dan diam memenuhi semua syarat", () => {
    const p = periksa(titik("terbuka_kanan"), "Right", titik("terbuka_kanan"), LEBAR, TINGGI);
    expect(p.semua).toBe(true);
    expect(p.sisi).toBe("kanan");
  });

  test("pesan untuk tangan mengepal dan tanpa tangan", () => {
    const p = periksa(titik("kepal_kanan"), "Right", titik("kepal_kanan"), LEBAR, TINGGI);
    expect(p.pesan).toBe("Buka semua jari");
    expect(periksa(null, null, null, LEBAR, TINGGI).pesan).toBe("Letakkan telapak di depan kamera");
  });
});

describe("kotak ROI", () => {
  test("sisinya selebar telapak", () => {
    const { sudut } = kotakRoi(titik("terbuka_kanan"), LEBAR, TINGGI);
    const sisi = Math.hypot(sudut[1][0] - sudut[0][0], sudut[1][1] - sudut[0][1]);
    expect(sisi).toBeCloseTo(Math.hypot(360 - 272, 300 - 312), 5);
  });

  test("kotak yang keluar dari video membuat syarat jarak gagal", () => {
    const keKanan = titik("terbuka_kanan").map((p) => ({ ...p, x: p.x + 300 / LEBAR }));
    expect(periksa(keKanan, "Right", keKanan, LEBAR, TINGGI).jarak).toBe(false);
  });
});

import { useEffect, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// Menjalankan MediaPipe Hands pada <video> di browser, setiap frame.
// Mengembalikan titik tangan terakhir, apakah detektor sudah siap, dan pesan galat kalau ada.
export function useHandTracker(videoRef) {
  const [tangan, setTangan] = useState(null); // { titik, label } atau null
  const [siap, setSiap] = useState(false);
  const [galat, setGalat] = useState(null);

  useEffect(() => {
    let aktif = true;
    let detektor = null;
    let idFrame = null;
    let waktuVideoTerakhir = -1;

    async function muat() {
      const opsi = {
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6,
      };
      try {
        // File-file ini disalin ke public/mediapipe oleh scripts/prepare-mediapipe.mjs
        const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        const model = "/mediapipe/hand_landmarker.task";
        try {
          detektor = await HandLandmarker.createFromOptions(fileset, {
            ...opsi,
            baseOptions: { modelAssetPath: model, delegate: "GPU" },
          });
        } catch {
          // Browser tanpa dukungan GPU: pakai CPU.
          detektor = await HandLandmarker.createFromOptions(fileset, {
            ...opsi,
            baseOptions: { modelAssetPath: model, delegate: "CPU" },
          });
        }
        if (!aktif) return;
        setSiap(true);
        idFrame = requestAnimationFrame(putaran);
      } catch (err) {
        if (aktif) setGalat(`Detektor tangan gagal dimuat: ${err.message}`);
      }
    }

    function putaran() {
      if (!aktif) return;
      const video = videoRef.current;
      const adaFrameBaru = video && video.readyState >= 2 && video.currentTime !== waktuVideoTerakhir;
      if (adaFrameBaru) {
        waktuVideoTerakhir = video.currentTime;
        const hasil = detektor.detectForVideo(video, performance.now());
        const titik = hasil.landmarks?.[0];
        const label = hasil.handedness?.[0]?.[0]?.categoryName;
        setTangan(titik ? { titik, label } : null);
      }
      idFrame = requestAnimationFrame(putaran);
    }

    muat();

    return () => {
      aktif = false;
      if (idFrame !== null) cancelAnimationFrame(idFrame);
      detektor?.close();
    };
  }, [videoRef]);

  return { tangan, siap, galat };
}

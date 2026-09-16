# PERCEPTION_LITE — P4-lite back-projection geométrica (veredicto: SHIP de stride — stride geométrico, no detector)

- Método: **geometría pura, cero ML**. Intrínsecos pinhole leídos del
  simulador en **solo lectura** (`sim/dual_so101_dinner.xml` vía
  `sim/load.py`: `cam_fovy=45.0°`, render 256×256 estándar del repo →
  `f=309.019px`, `cx=cy=127.5`). Depth del renderer MuJoCo (EGL, en memoria;
  ningún fichero de imagen leído ni escrito). **Sin detector entrenado,
  sin VLM/LLM, sin entrenamiento.**
- Máscara: segmentación del simulador (`mug_geom`, 357 px) usada como
  **STAND-IN de detector DECLARADO**: mide solo la cadena
  intrínsecos+back-projection, jamás se presenta como detección.
- Spawn: defecto del XML (`mug=[-0.12,0.02,0.794]`), **sin bundles de
  seeds** (0–119 no tocadas ni leídas salvo formatos). Cámara `overhead`.
- Código: `out/grounding/perception_lite.py`. Log: `out/grounding/perception_run.log` (exit 0).
- Time-box: 1 implementación directa + 1 fix de ruta; sin iteración extra.

## Números propios (reales, del log)

- `err_est_xy=2.77mm` vs `baseline ingenuo centro-mesa err_base_xy=130.00mm` — 2.77mm xy-only, oracle mask, sesgo Z 29mm; stride geométrico, no detector
- `reproj_selfcheck=0.81px` (re-proyección del GT contra centroide de máscara, ≤3px exigido)
- `err_est_3d=29.41mm` (sesgo Z honesto — 2.77mm xy-only, oracle mask, sesgo Z 29mm; stride geométrico, no detector, ver abajo)

## Veredicto: SHIP de stride (stride geométrico, no detector; 2.77mm xy-only, oracle mask, sesgo Z 29mm)

Stride superado: error XY comparable o mejor que el baseline ingenuo
(2.77mm ≪ 130mm — 2.77mm xy-only, oracle mask, sesgo Z 29mm; stride geométrico, no detector)
con autochequeo geométrico consistente (0.81px).
Alcance honesto: valida la cadena geométrica con asociación oracle, no un
perceptor.

## Honest-negative / límites (se archivan, no se persiguen aquí)

1. El sesgo Z (~29mm) es esperado: el centroide back-proyectado es de la
   **superficie visible** (tapa del mug, ~0.823) mientras el GT es el
   **centro del cuerpo** (0.794). Para control haría falta completar forma
   o prior de altura — fuera del stride P4-lite.
2. Asociación oracle: con un detector real el error sería mayor; este
   número es cota inferior de la cadena geométrica, no rendimiento de
   percepción.
3. Un solo spawn por defecto + una sola cámara (`overhead`); sin barrido
   de poses, sin oclusiones por brazos en movimiento, sin ruido de depth.
4. Sin claim de LLM/VLM/detector entrenado en ningún punto del informe.

#!/usr/bin/env python3
"""
RPi5 + Hailo-8: Object Detection via Camera CSI-2 con GStreamer.

Filtra le bounding box: vengono disegnate SOLO le detection con label "person".
Mostra stream video con bounding box persone, conteggio, FPS e temperatura SoC RPi5.

I servomotori (Hardware PWM) seguono automaticamente le persone rilevate:
la telecamera punta sempre verso la posizione media di tutte le persone nel frame.
"""

import os
import sys
import time
import threading
from datetime import datetime

try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GLib
    Gst.init(None)
except ImportError:
    print("Modulo 'gi' non trovato. Esegui: sudo apt install python3-gi python3-gst-1.0")
    sys.exit(1)

try:
    import hailo
    HAILO_PYTHON_AVAILABLE = True
except ImportError:
    HAILO_PYTHON_AVAILABLE = False
    print("⚠️  Modulo 'hailo' (Python bindings) non trovato.")
    print("    Il filtro 'person-only' sarà disabilitato: verranno mostrate tutte le detection.")
    print("    Per abilitarlo, attiva l'ambiente TAPPAS: source /opt/hailo/tappas/setup_env.sh\n")

try:
    from rpi_hardware_pwm import HardwarePWM
    PWM_AVAILABLE = True
except ImportError:
    PWM_AVAILABLE = False
    print("⚠️  Modulo 'rpi_hardware_pwm' non trovato.")
    print("    Il tracking servo sarà disabilitato.")
    print("    Per abilitarlo: pip3 install rpi-hardware-pwm\n")

# --- CONFIGURAZIONI CAMERA ---
HEF_PATH          = "/home/olivola/Desktop/Tirocinio/Modello/yolov8n.hef"
POST_PROCESS_SO   = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
POST_PROCESS_JSON = "/home/olivola/Desktop/Tirocinio/Raspbarry/person_only_labels.json"

CAM_WIDTH       = 1280
CAM_HEIGHT      = 720
CAM_FPS         = 30

THERMAL_ZONE    = "/sys/class/thermal/thermal_zone0/temp"
TEMP_INTERVALLO = 2   # secondi tra un aggiornamento HUD e l'altro

TARGET_LABEL    = "person"   # YOLOv8 COCO: "person" = classe 0

# --- CONFIGURAZIONI SERVO PWM ---
# Pin 12 → PWM0_CH0 → asse Y (tilt verticale)
# Pin 13 → PWM0_CH1 → asse X (pan orizzontale)
PWM_HZ   = 50
PWM_CHIP = 0

PAN_MIN,  PAN_MAX  = -1.0, +1.0   # limiti asse X
TILT_MIN, TILT_MAX = -1.0, +0.5   # limiti asse Y (asimmetrici per vincoli fisici)

# --- CONFIGURAZIONI PID ---
# Guadagni del controller proporzionale-integrale-derivativo.
# Aumenta KP per reazione più rapida; abbassa se i servo oscillano.
# KI compensa errori stazionari persistenti.
# KD smorza le oscillazioni (damping).
PID_KP = 0.15
PID_KI = 0.02
PID_KD = 0.05

# Zona morta: errori più piccoli di questa soglia (in frazione del frame)
# vengono ignorati per evitare micro-vibrazioni dei servo.
PID_DEADZONE = 0.03   # 3% del lato del frame

# Fattore di smoothing EMA (Exponential Moving Average) sul comando servo.
# 0.0 = servo bloccato, 1.0 = nessun smoothing (salto diretto).
# Valori bassi → movimento più fluido ma più lento a raggiungere il target.
SERVO_ALPHA = 0.12


# --- STATO CONDIVISO ---
_fps_lock        = threading.Lock()
_fps_frame_count = 0
_fps_last_count  = 0
_fps_last_time   = 0.0
_person_count    = 0

# Posizione corrente dei servo (normalizzata -1..+1)
_servo_lock = threading.Lock()
_pan        = 0.0
_tilt       = 0.0

# Stato interno PID (un set per asse)
_pid_integral_x   = 0.0
_pid_integral_y   = 0.0
_pid_error_prev_x = 0.0
_pid_error_prev_y = 0.0
_pid_last_time    = 0.0

# Oggetti HardwarePWM (inizializzati in avvia())
_rotore_x: "HardwarePWM | None" = None
_rotore_y: "HardwarePWM | None" = None


# ─── SERVO: CONVERSIONE POSIZIONE → DUTY CYCLE ────────────────────────────────

def _imposta_servo(hwpwm: "HardwarePWM", valore: float) -> None:
    """
    Mappa valore in [-1.0, +1.0] nel duty cycle corrispondente:
      -1.0 → 2.5%  (impulso 0.5 ms)
       0.0 → 7.5%  (impulso 1.5 ms, centro)
      +1.0 → 12.5% (impulso 2.5 ms)
    """
    duty = 7.5 + valore * 5.0
    hwpwm.change_duty_cycle(duty)


# ─── CONTROLLER PID ───────────────────────────────────────────────────────────

def _pid_update(error_x: float, error_y: float) -> None:
    """
    Aggiorna la posizione dei servo tramite un controller PID.

    error_x, error_y: errore normalizzato in [-0.5, +0.5] rispetto al centro frame.
    Positivo → target a destra/basso rispetto al centro.
    """
    global _pan, _tilt
    global _pid_integral_x, _pid_integral_y
    global _pid_error_prev_x, _pid_error_prev_y, _pid_last_time

    if _rotore_x is None or _rotore_y is None:
        return

    now = time.monotonic()
    dt  = now - _pid_last_time
    if dt <= 0 or dt > 1.0:
        # Prima chiamata o gap troppo lungo: reset derivata
        dt = 0.033  # assume ~30fps
    _pid_last_time = now

    # Zona morta: ignora piccoli errori per evitare vibrazioni
    if abs(error_x) < PID_DEADZONE:
        error_x = 0.0
        _pid_integral_x = 0.0
    if abs(error_y) < PID_DEADZONE:
        error_y = 0.0
        _pid_integral_y = 0.0

    # Aggiorna integrali con anti-windup (clamp)
    _pid_integral_x = max(-0.5, min(0.5, _pid_integral_x + error_x * dt))
    _pid_integral_y = max(-0.5, min(0.5, _pid_integral_y + error_y * dt))

    # Derivata
    deriv_x = (error_x - _pid_error_prev_x) / dt
    deriv_y = (error_y - _pid_error_prev_y) / dt
    _pid_error_prev_x = error_x
    _pid_error_prev_y = error_y

    # Output PID
    out_x = PID_KP * error_x + PID_KI * _pid_integral_x + PID_KD * deriv_x
    out_y = PID_KP * error_y + PID_KI * _pid_integral_y + PID_KD * deriv_y

    with _servo_lock:
        # Il pan segue l'errore X invertito: se la persona è a destra (error_x > 0),
        # il servo deve ruotare a destra → sottraiamo out_x
        target_pan  = max(PAN_MIN,  min(PAN_MAX,  _pan  - out_x))
        # Il tilt segue l'errore Y: se la persona è in basso (error_y > 0),
        # il tilt scende (diminuisce _tilt)
        target_tilt = max(TILT_MIN, min(TILT_MAX, _tilt - out_y))
        # EMA: blend graduale verso il target (SERVO_ALPHA = velocità di inseguimento)
        _pan  = _pan  + SERVO_ALPHA * (target_pan  - _pan)
        _tilt = _tilt + SERVO_ALPHA * (target_tilt - _tilt)

    _imposta_servo(_rotore_x, _pan)
    _imposta_servo(_rotore_y, _tilt)


# ─── PROBE FILTRO "PERSON ONLY" + TRACKING ────────────────────────────────────

def _filtro_person_probe(pad, info) -> Gst.PadProbeReturn:
    """
    Pad probe sul sink di hailooverlay.

    1. Filtra le detection: mantiene solo label == TARGET_LABEL.
    2. Calcola il centro medio di tutte le persone rilevate.
    3. Alimenta il PID per muovere i servo verso quel punto.
    """
    global _person_count

    if not HAILO_PYTHON_AVAILABLE:
        return Gst.PadProbeReturn.OK

    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    try:
        roi = hailo.get_roi_from_buffer(buf)

        # list() snapshot necessario: roi.remove_object() modifica la lista
        # interna e causerebbe salti durante l'iterazione.
        detections = list(roi.get_objects_typed(hailo.HAILO_DETECTION))

        persons = []
        for det in detections:
            if det.get_label().lower() != TARGET_LABEL.lower():
                roi.remove_object(det)
            else:
                persons.append(det)

        with _fps_lock:
            _person_count = len(persons)

        # ── Tracking: calcola il centro medio di tutte le persone ──────────
        if persons and PWM_AVAILABLE:
            # Ogni bbox ha coordinate normalizzate in [0, 1]
            cx_sum = 0.0
            cy_sum = 0.0
            for det in persons:
                bbox  = det.get_bbox()
                cx_sum += bbox.xmin() + bbox.width()  / 2.0
                cy_sum += bbox.ymin() + bbox.height() / 2.0

            n    = len(persons)
            cx   = cx_sum / n   # centro X medio, in [0, 1]
            cy   = cy_sum / n   # centro Y medio, in [0, 1]

            # Errore rispetto al centro del frame (0.5, 0.5)
            error_x = cx - 0.5   # positivo → target a destra
            error_y = cy - 0.5   # positivo → target in basso

            _pid_update(error_x, error_y)

    except Exception:
        pass

    return Gst.PadProbeReturn.OK


# ─── CONTATORE FPS ────────────────────────────────────────────────────────────

def _on_frame_probe(pad, info) -> Gst.PadProbeReturn:
    """Probe sul pad src di textoverlay: incrementa il contatore frame per il calcolo FPS."""
    global _fps_frame_count
    with _fps_lock:
        _fps_frame_count += 1
    return Gst.PadProbeReturn.OK


# ─── TEMPERATURA E HUD ────────────────────────────────────────────────────────

def _leggi_temp_rpi5() -> float | None:
    """Legge la temperatura del SoC RPi5 dal file di sistema."""
    try:
        with open(THERMAL_ZONE, "r", encoding="utf-8") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _testo_hud(fps: float, n_persone: int) -> str:
    """Compone la stringa dell'overlay HUD."""
    ora  = datetime.now().strftime("%H:%M:%S")
    t    = _leggi_temp_rpi5()
    temp = f"{t:.1f}C" if t is not None else "N/A"
    with _servo_lock:
        pan_val  = _pan
        tilt_val = _tilt
    servo_str = f"Pan:{pan_val:+.2f} Tilt:{tilt_val:+.2f}" if PWM_AVAILABLE else "Servo: N/A"
    return (
        f"[{ora}]  RPi5: {temp}  |  FPS: {fps:.1f}  |  "
        f"YOLOv8n [Hailo-8]  |  Persone: {n_persone}  |  {servo_str}"
    )


# ─── PIPELINE GSTREAMER ───────────────────────────────────────────────────────

def _build_pipeline_str() -> str:
    """
    Costruisce la stringa della pipeline GStreamer.

    Sorgente  : libcamerasrc  (fotocamera CSI-2)
    Elaboraz. : hailonet → hailofilter (config person-only) → hailooverlay (+ probe filtro+tracking)
    Output    : autovideosink con overlay testuale HUD
    """
    return (
        f"libcamerasrc ! "
        f"video/x-raw,width={CAM_WIDTH},height={CAM_HEIGHT},"
        f"framerate={CAM_FPS}/1,format=RGB ! "

        f"queue max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
        f"videoconvert ! videoscale ! "
        f"video/x-raw,format=RGB,pixel-aspect-ratio=1/1 ! "

        f"queue max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
        f"hailonet hef-path={HEF_PATH} scheduling-algorithm=1 ! "

        f"queue max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
        f"hailofilter so-path={POST_PROCESS_SO} "
        f"config-path={POST_PROCESS_JSON} qos=false ! "

        # name=overlay necessario per agganciare il probe filtro+tracking sul sink pad
        f"hailooverlay name=overlay ! videoconvert ! "

        f"textoverlay name=hud valignment=top halignment=left "
        f"font-desc=\"monospace Bold 14\" shaded-background=true text=\"Avvio...\" ! "
        f"autovideosink sync=false"
    )


# ─── AVVIO ────────────────────────────────────────────────────────────────────

def avvia() -> None:
    """Costruisce ed esegue la pipeline di rilevamento con camera CSI-2 e tracking servo."""
    global _rotore_x, _rotore_y, _pid_last_time

    print("\n🚀 Avvio RPi5 + Hailo-8 Object Detection — Camera CSI-2 (libcamerasrc)")
    print(f"   Filtro attivo:  {'✅ PERSON ONLY' if HAILO_PYTHON_AVAILABLE else '⚠️  disabilitato (binding Python Hailo assenti)'}")
    print(f"   Tracking servo: {'✅ abilitato' if PWM_AVAILABLE else '⚠️  disabilitato (rpi_hardware_pwm non trovato)'}\n")

    if not os.path.exists(HEF_PATH):
        print(f"❌ Modello non trovato: {HEF_PATH}")
        return
    if not os.path.exists(POST_PROCESS_SO):
        print(f"❌ Post-process library non trovata: {POST_PROCESS_SO}")
        return

    # ── Inizializzazione servo ─────────────────────────────────────────────────
    if PWM_AVAILABLE:
        try:
            _rotore_y = HardwarePWM(pwm_channel=0, hz=PWM_HZ, chip=PWM_CHIP)  # Pin 12, tilt
            _rotore_x = HardwarePWM(pwm_channel=1, hz=PWM_HZ, chip=PWM_CHIP)  # Pin 13, pan
            _rotore_x.start(7.5)   # centra pan
            _rotore_y.start(7.5)   # centra tilt
            _pid_last_time = time.monotonic()
            print("✅ Servo inizializzati e centrati.")
        except Exception as e:
            print(f"⚠️  Impossibile inizializzare i servo: {e}")
            print("    Il tracking sarà disabilitato per questa sessione.")
            _rotore_x = None
            _rotore_y = None

    pipeline_str = _build_pipeline_str()
    print(f"Pipeline:\n  {pipeline_str[:120]}...\n")

    pipeline = Gst.parse_launch(pipeline_str)
    if not pipeline:
        print("❌ Errore nella creazione della pipeline.")
        return

    # Probe filtro + tracking: agisce PRIMA che hailooverlay disegni i bounding box
    overlay_elem = pipeline.get_by_name("overlay")
    if overlay_elem and HAILO_PYTHON_AVAILABLE:
        sink_pad = overlay_elem.get_static_pad("sink")
        if sink_pad:
            sink_pad.add_probe(Gst.PadProbeType.BUFFER, _filtro_person_probe)
            print("✅ Probe filtro+tracking agganciato su hailooverlay:sink")
        else:
            print("⚠️  Pad sink di hailooverlay non trovato — filtro disabilitato")
    elif not HAILO_PYTHON_AVAILABLE:
        print("⚠️  Filtro non agganciato: binding Python Hailo non disponibili")
    else:
        print("⚠️  Elemento 'overlay' non trovato nella pipeline — filtro disabilitato")

    # Probe FPS
    hud = pipeline.get_by_name("hud")
    if hud:
        src_pad = hud.get_static_pad("src")
        if src_pad:
            src_pad.add_probe(Gst.PadProbeType.BUFFER, _on_frame_probe)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("❌ Impossibile avviare la pipeline.")
        print("   Controlla che la fotocamera CSI-2 sia collegata e riconosciuta.")
        print("   Prova prima: rpicam-hello")
        bus_tmp = pipeline.get_bus()
        msg = bus_tmp.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR)
        if msg:
            err, debug = msg.parse_error()
            print(f"   Dettaglio errore: {err.message}")
            if debug:
                print(f"   Debug info: {debug}")
        pipeline.set_state(Gst.State.NULL)
        _ferma_servo()
        return

    print("\n✅ Telecamera e Hailo avviati con successo!")
    print("   Rilevamento in corso. Premi CTRL+C per uscire.\n")

    loop = GLib.MainLoop()

    def _aggiorna_hud() -> bool:
        global _fps_last_count, _fps_last_time
        now = time.monotonic()
        with _fps_lock:
            count    = _fps_frame_count
            n_person = _person_count
        elapsed = now - _fps_last_time
        fps = (count - _fps_last_count) / elapsed if elapsed > 0 else 0.0
        _fps_last_count = count
        _fps_last_time  = now
        if hud:
            hud.set_property("text", _testo_hud(fps, n_person))
        return True

    _fps_last_time = time.monotonic()
    GLib.timeout_add_seconds(TEMP_INTERVALLO, _aggiorna_hud)
    _aggiorna_hud()

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_message(bus, msg) -> None:
        if msg.type == Gst.MessageType.EOS:
            print("\nFine dello stream video.")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"\n❌ Errore GStreamer: {err.message}")
            if debug:
                print(f"   Debug info: {debug}")
            loop.quit()
        elif msg.type == Gst.MessageType.WARNING:
            wrn, _ = msg.parse_warning()
            if "position" not in wrn.message.lower():
                print(f"⚠️  {wrn.message}")

    bus.connect("message", _on_message)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterruzione da tastiera rilevata...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        _ferma_servo()
        print("Pipeline chiusa. Uscita completata.")


def _ferma_servo() -> None:
    """Ferma i canali PWM in sicurezza."""
    global _rotore_x, _rotore_y
    for nome, rotore in [("X (pan)", _rotore_x), ("Y (tilt)", _rotore_y)]:
        if rotore is not None:
            try:
                rotore.stop()
                print(f"   Servo {nome} fermato.")
            except Exception as e:
                print(f"   ⚠️  Errore stop servo {nome}: {e}")
    _rotore_x = None
    _rotore_y = None


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Punto di entrata principale.

    Uso:
        python3 csi_hailo_detect.py
    """
    avvia()


if __name__ == "__main__":
    main()

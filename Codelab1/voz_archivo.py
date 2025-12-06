import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile, os

SRATE = 16000     # tasa de muestreo
DUR = 5           # segundos

print("Grabando... habla ahora!")
audio = sd.rec(int(DUR*SRATE), samplerate=SRATE, channels=1, dtype='int16')
sd.wait()
print("Listo, procesando...")

# guarda a WAV temporal
tmp_wav = tempfile.mktemp(suffix=".wav")
write(tmp_wav, SRATE, audio)

# reconoce con SpeechRecognition
r = sr.Recognizer()
with sr.AudioFile(tmp_wav) as source:
    data = r.record(source)

try:
    texto = r.recognize_google(data, language="es-ES")
    cmd = texto.lower()
    print("Dijiste:", texto)
    if "hola" in cmd:
        print("¡Hola, bienvenido al curso!")
    elif "quiero trabajar" in cmd:
        import webbrowser
        webbrowser.open("https://co.linkedin.com/")
    elif "hora" in cmd:
        from datetime import datetime
        print("Hora actual:", datetime.now().strftime("%H:%M"))
    elif "feliz" in cmd:
        import webbrowser
        print(":3")
    elif "extremoduro" in cmd:
        import webbrowser
        webbrowser.open("https://youtu.be/43S_qfT6vpo?si=94EIvv8S7BsBOwSB")
    elif "traducir" in cmd:
        import asyncio
        from googletrans import Translator
        translator= Translator()
        result = asyncio.run(translator.translate(cmd, dest = 'en'))
        print(result.text)

    else:
        print("Comando no reconocido.")
except sr.UnknownValueError:
    print("No se entendió el audio.")
except sr.RequestError as e:
    print("Error:", e)
finally:
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
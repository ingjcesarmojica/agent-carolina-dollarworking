import os
import io
import asyncio
import base64
import re
import json
import tempfile
import threading
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import edge_tts
import google.generativeai as genai
from dotenv import load_dotenv

try:
    from rag import search_knowledge, add_pdf, list_documents, delete_document

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    GEMINI_CONFIGURED = True
else:
    gemini_model = None
    GEMINI_CONFIGURED = False
    app.logger.warning(
        "GEMINI_API_KEY no configurada - chat usar solo respuestas hardcoded"
    )

TTS_VOICE = os.environ.get("TTS_VOICE", "es-US-PalomaNeural")


async def generate_edge_tts(text, voice=None):
    if voice is None:
        voice = TTS_VOICE
    communicate = edge_tts.Communicate(text, voice)
    tmp_path = os.path.join(os.path.dirname(__file__), "tmp_audio.mp3")
    await communicate.save(tmp_path)
    with open(tmp_path, "rb") as f:
        audio_data = f.read()
    os.remove(tmp_path)
    return base64.b64encode(audio_data).decode("utf-8")


@app.before_request
def log_config():
    app.logger.info(f"Gemini configured: {GEMINI_CONFIGURED}")
    app.logger.info(f"TTS Voice: {TTS_VOICE}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/speak", methods=["POST"])
def speak_text():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        app.logger.info(f"Generando audio con edge-tts: {text[:50]}...")
        audio_content = asyncio.run(generate_edge_tts(text))

        return jsonify(
            {
                "audioContent": audio_content,
                "audioUrl": f"data:audio/mp3;base64,{audio_content}",
                "useBrowserTTS": False,
                "engine": "edge-tts",
            }
        )

    except Exception as e:
        app.logger.error(f"Error en edge-tts: {str(e)}")
        return jsonify(
            {
                "audioContent": None,
                "audioUrl": None,
                "useBrowserTTS": True,
                "text": text,
                "error": str(e),
            }
        )


def gemini_response(user_message, context=""):
    if not GEMINI_CONFIGURED or gemini_model is None:
        return None
    try:
        system_prompt = """Eres Claudia García, abogada virtual especializada en Derecho Laboral de TusAbogados.com.

## Tu personalidad
- Eres una abogada laboralista con experiencia.
- Hablas con profesionalismo y calidez, como lo haría un abogado real.
- Usas terminología legal cuando es apropiado, pero la explicas en lenguaje sencillo.
- Transmites confianza, seguridad y empatía.
- Ejemplos de expresiones naturales: "Entiendo perfectamente su situación", "Esto es algo que manejamos con frecuencia", "Le comento que en estos casos...", "Es importante que sepa que...", "Procederemos a..."

## Reglas
- Responde en máximo 2-3 oraciones.
- Si te preguntan algo de derecho laboral, responde con precisión legal pero explicando en lenguaje simple.
- Usa términos como: despido injustificado, justa causa, liquidación, prestaciones sociales, indemnización, conciliación, juzgado laboral, derecho laboral.
- Siempre orienta pero NO das asesoría legal definitiva, eso lo hace el abogado humano.
- Nunca uses expresiones informales como "genial", "perfecto", "listo", "dale". Usa: "Entiendo", "Comprendo", "Procederé a", "Le comento que"."""

        rag_context = ""
        if RAG_AVAILABLE:
            try:
                docs = search_knowledge(user_message, n_results=3)
                if docs:
                    rag_parts = []
                    for d in docs:
                        rag_parts.append(f"[Fuente: {d['source']}]\n{d['text']}")
                    rag_context = (
                        "\n\n## Base de conocimiento (usa esta información si es relevante):\n"
                        + "\n---\n".join(rag_parts)
                    )
            except Exception:
                pass

        prompt = f"""{system_prompt}{rag_context}

Contexto: {context}
Usuario: {user_message}"""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        app.logger.error(f"Error Gemini: {str(e)}")
        return None


def validate_name(name):
    if not name or len(name.strip()) < 2:
        return (
            False,
            "Por favor, indíqueme su nombre completo para proceder con la cita.",
        )
    if re.match(r"^[\d\s]+$", name.strip()):
        return (
            False,
            "El nombre ingresado no parece válido. Por favor, indíqueme su nombre completo.",
        )
    return True, name.strip()


def validate_subtype(subtype):
    if not subtype or len(subtype.strip()) < 3:
        return (
            False,
            "¿Qué tipo de situación laboral está atrayendo? Por ejemplo, despido injustificado, acoso laboral, impago de prestaciones...",
        )
    return True, subtype.strip()


def validate_description(desc):
    if not desc or len(desc.strip()) < 5:
        return (
            False,
            "Le agradecería que me describa brevemente los hechos de su caso: fechas, personas involucradas y circunstancias.",
        )
    return True, desc.strip()


def validate_email(email):
    if not email:
        return (
            False,
            "¿Cuál es su correo electrónico? Lo necesito para enviarle la confirmación de la cita.",
        )
    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        return (
            False,
            "El correo electrónico ingresado no tiene un formato válido. Por favor, verifíquelo e ingréselo nuevamente (ejemplo: nombre@correo.com).",
        )
    return True, email.strip()


def validate_phone(phone):
    if not phone:
        return False, "¿Cuál es su número de teléfono de contacto?"
    digits = re.sub(r"[^0-9]", "", phone)
    if len(digits) < 7 or len(digits) > 15:
        return (
            False,
            "El número de teléfono ingresado no parece correcto. Por favor, verifíquelo e ingréselo sin espacios ni guiones (ejemplo: 3001234567).",
        )
    return True, digits


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "No message provided"}), 400

        message_lower = message.lower().strip()

        is_greeting = any(
            word in message_lower
            for word in [
                "hola",
                "buenos días",
                "buenas tardes",
                "saludos",
                "buenas",
                "buenos",
                "iniciar",
                "empezar",
            ]
        )

        is_farewell = any(
            word in message_lower
            for word in [
                "gracias",
                "adiós",
                "chao",
                "hasta luego",
                "no gracias",
                "eso es todo",
            ]
        )

        is_repeat = any(
            word in message_lower
            for word in ["repetir", "repita", "no entendí", "cómo", "que dijiste"]
        )

        is_confirm = any(
            word in message_lower
            for word in [
                "sí",
                "si",
                "ok",
                "de acuerdo",
                "confirmo",
                "dale",
                "perfecto",
                "va",
            ]
        )

        is_reject = any(
            word in message_lower
            for word in ["no", "no me viene", "otro horario", "otra hora", "no puedo"]
        )

        if is_greeting:
            for attr in [
                "user_name",
                "user_email",
                "user_phone",
                "case_description",
                "case_subtype",
                "appointment_time",
            ]:
                if hasattr(chat, attr):
                    delattr(chat, attr)
            response = "Buenos días. Soy Claudia García, abogada laboralista de TusAbogados.com. Le comento que estamos aquí para asistirle con su caso. Para iniciar con su consulta, ¿podría indicarme su nombre completo?"
            return jsonify({"response": response, "end_call": False})

        if is_farewell:
            name = getattr(chat, "user_name", "")
            if hasattr(chat, "appointment_time"):
                response = f"Entendido, {name}. Le confirmo que un abogado laboralista se pondrá en contacto con usted en la fecha acordada. Cualquier consulta adicional, no dude en escribirnos. Saludos cordiales."
            else:
                response = f"Entendido, {name}. Un abogado laboralista se comunicará con usted a la brevedad. Cualquier consulta adicional, no dude en escribirnos. Saludos cordiales."
            return jsonify({"response": response, "end_call": True})

        if is_repeat:
            if not hasattr(chat, "user_name"):
                response = (
                    "Por favor, indíqueme su nombre completo para proceder con la cita."
                )
            elif not hasattr(chat, "case_subtype"):
                response = "¿Qué tipo de situación laboral está atrayendo? Por ejemplo, despido injustificado, acoso laboral, impago de prestaciones..."
            elif not hasattr(chat, "case_description"):
                response = "Le agradecería que me describa brevemente los hechos de su caso: fechas, personas involucradas y circunstancias."
            elif not hasattr(chat, "user_email"):
                response = "¿Cuál es su correo electrónico? Lo necesito para enviarle la confirmación de la cita."
            elif not hasattr(chat, "user_phone"):
                response = "¿Cuál es su número de teléfono de contacto?"
            elif not hasattr(chat, "appointment_time"):
                response = "¿Le viene bien el Lunes 29 de septiembre a las 10:30 a.m.?"
            else:
                response = "¿Hay algo más en lo que pueda asistirle?"
            return jsonify({"response": response, "end_call": False})

        if (
            hasattr(chat, "appointment_time")
            and not is_confirm
            and not is_farewell
            and len(message.strip()) > 3
        ):
            context = f"Usuario: {getattr(chat, 'user_name', '')}, caso: {getattr(chat, 'case_subtype', '')}, cita ya agendada. Responde con terminología legal profesional."
            gemini_resp = gemini_response(message, context=context)
            if gemini_resp:
                response = f"Le comento que {gemini_resp}\n\n¿Hay algo más en lo que pueda asistirle?"
            else:
                response = f"Entendido, {getattr(chat, 'user_name', '')}. He registrado su consulta. Un abogado laboralista le ampliará la información cuando se contacte con usted.\n\n¿Hay algo más en lo que pueda asistirle?"
            return jsonify({"response": response, "end_call": False})

        if not hasattr(chat, "appointment_time") and is_confirm:
            chat.appointment_time = "Lunes 29 de Septiembre - 10:30 am"
            name = getattr(chat, "user_name", "")
            email = getattr(chat, "user_email", "")
            phone = getattr(chat, "user_phone", "")
            subtype = getattr(chat, "case_subtype", "")
            response = f"""Procederé a confirmar su cita.

📅 Fecha: Lunes 29 de septiembre - 10:30 a.m.
📧 Correo de confirmación: {email}
📱 Teléfono de contacto: {phone}

He analizado su caso de {subtype}. Le comento que, si el monto supera los 10 millones de pesos, no hay costo inicial: solo se aplica un honoratorio del 10% en caso de éxito.

Un abogado laboralista se comunicará con usted a la brevedad. ¿Hay algo más en lo que pueda asistirle?"""
            return jsonify({"response": response, "end_call": False})

        if not hasattr(chat, "appointment_time") and is_reject:
            response = "Entiendo perfectamente. En ese caso, ¿le gustaría que le pongamos en contacto directamente con uno de nuestros abogados laboralistas? Ellos podrán atender su caso de forma personalizada."
            return jsonify({"response": response, "end_call": False})

        if not hasattr(chat, "appointment_time") and any(
            w in message_lower for w in ["miércoles", "miercoles", "tarde"]
        ):
            chat.appointment_time = "Miércoles 1 de Octubre - 3:30 pm"
            name = getattr(chat, "user_name", "")
            email = getattr(chat, "user_email", "")
            phone = getattr(chat, "user_phone", "")
            subtype = getattr(chat, "case_subtype", "")
            response = f"""Queda registrada su cita.

📅 Fecha: Miércoles 1 de octubre - 3:30 p.m.
📧 Correo de confirmación: {email}
📱 Teléfono de contacto: {phone}

He revisado su caso de {subtype}. Un abogado laboralista se comunicará con usted en la fecha acordada.

¿Hay algo más en lo que pueda asistirle?"""
            return jsonify({"response": response, "end_call": False})

        if not hasattr(chat, "user_name"):
            valid, result = validate_name(message)
            if valid:
                chat.user_name = result
                response = f"Mucho gusto, {result}. Para orientarle correctamente, ¿podría indicarme qué tipo de situación laboral está atrayendo? Por ejemplo, despido injustificado, acoso laboral, impago de prestaciones..."
            else:
                response = result
            return jsonify({"response": response, "end_call": False})

        if not hasattr(chat, "case_subtype"):
            valid, result = validate_subtype(message)
            if valid:
                chat.case_subtype = result
                response = "Comprendo. Le agradecería que me describa brevemente los hechos: fechas, personas involucradas y circunstancias del caso."
            else:
                response = result
            return jsonify({"response": response, "end_call": False})

        if not hasattr(chat, "case_description"):
            valid, result = validate_description(message)
            if valid:
                chat.case_description = result
                response = f"Agradezco la información, {getattr(chat, 'user_name', '')}. Para proceder con el agendamiento de su cita y enviarle la confirmación, ¿cuál es su correo electrónico?"
            else:
                response = result
            return jsonify({"response": response, "end_call": False})

        if not hasattr(chat, "user_email"):
            valid, result = validate_email(message)
            if valid:
                chat.user_email = result
                response = "Perfecto. Ahora necesito tu número de celular para poder contactarte."
            else:
                response = result
            return jsonify({"response": response, "end_call": False})

        if not hasattr(chat, "user_phone"):
            valid, result = validate_phone(message)
            if valid:
                chat.user_phone = result
                response = f"Excelente, {getattr(chat, 'user_name', '')}. Ya cuento con toda la información necesaria. Analizando su caso de {getattr(chat, 'case_subtype', '')}, le recomendaría una cita con uno de nuestros abogados laboralistas. ¿Le viene bien el Lunes 29 de septiembre a las 10:30 a.m.?"
            else:
                response = result
            return jsonify({"response": response, "end_call": False})

        response = "¿Hay algo más en lo que pueda ayudarte?"
        return jsonify({"response": response, "end_call": False})

    except Exception as e:
        app.logger.error(f"Exception in chat: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/knowledge/upload", methods=["POST"])
def upload_knowledge():
    if not RAG_AVAILABLE:
        return jsonify(
            {"error": "Módulo RAG no disponible. Verifique dependencias."}
        ), 500
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo."}), 400
    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Solo se permiten archivos PDF."}), 400

    # Check file size (max 5MB to prevent OOM on Render free tier)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Seek back to start
    max_size = 5 * 1024 * 1024  # 5MB
    if file_size > max_size:
        return jsonify(
            {
                "error": f"El archivo excede el límite de 5MB. Tamaño actual: {file_size // (1024 * 1024)}MB"
            }
        ), 400

    try:
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, file.filename)
        file.save(tmp_path)
        app.logger.info(f"PDF guardado temporalmente: {tmp_path}")

        num_chunks, msg = add_pdf(tmp_path)
        app.logger.info(f"Resultado add_pdf: {msg}")

        # Cleanup
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

        if num_chunks == 0:
            return jsonify({"error": msg}), 400
        return jsonify({"message": msg, "chunks": num_chunks})
    except Exception as e:
        app.logger.error(f"Error uploading PDF: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error al procesar el PDF: {str(e)}"}), 500


@app.route("/api/knowledge/documents", methods=["GET"])
def list_knowledge():
    if not RAG_AVAILABLE:
        return jsonify({"documents": [], "rag_available": False})
    docs = list_documents()
    return jsonify({"documents": docs, "rag_available": True})


@app.route("/api/knowledge/delete", methods=["POST"])
def delete_knowledge():
    if not RAG_AVAILABLE:
        return jsonify({"error": "Módulo RAG no disponible."}), 500
    data = request.json
    source = data.get("source", "")
    if not source:
        return jsonify({"error": "Nombre del documento no proporcionado."}), 400
    success, msg = delete_document(source)
    if success:
        return jsonify({"message": msg})
    return jsonify({"error": msg}), 404


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "gemini_configured": GEMINI_CONFIGURED,
            "tts_voice": TTS_VOICE,
            "service": f"edge-tts ({TTS_VOICE}) + Gemini Flash"
            if GEMINI_CONFIGURED
            else f"edge-tts ({TTS_VOICE}) + Hardcoded",
        }
    )


@app.route("/api/test-embedding", methods=["GET"])
def test_embedding():
    """Test endpoint to check if Gemini embeddings work."""
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return jsonify({"error": "GEMINI_API_KEY no configurada"}), 500
        genai.configure(api_key=api_key)
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content="Test de embedding",
            output_dimensionality=768,
        )
        return jsonify(
            {
                "status": "ok",
                "dimension": len(result["embedding"]),
                "first_5_values": result["embedding"][:5],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/voices", methods=["GET"])
def list_voices():
    voices = [
        {
            "id": "es-US-PalomaNeural",
            "name": "Paloma",
            "gender": "Femenina",
            "region": "Estados Unidos (español)",
            "recommended": True,
        },
        {
            "id": "es-MX-DaliaNeural",
            "name": "Dalia",
            "gender": "Femenina",
            "region": "México",
        },
        {
            "id": "es-MX-JorgeNeural",
            "name": "Jorge",
            "gender": "Masculino",
            "region": "México",
        },
        {
            "id": "es-ES-ElviraNeural",
            "name": "Elvira",
            "gender": "Femenina",
            "region": "España",
        },
        {
            "id": "es-ES-AlvaroNeural",
            "name": "Álvaro",
            "gender": "Masculino",
            "region": "España",
        },
    ]
    return jsonify({"voices": voices, "current": TTS_VOICE})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

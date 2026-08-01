import os
import asyncio
import base64
import re
import json
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
        "GEMINI_API_KEY no configurada - chat usando respuestas hardcoded"
    )

TTS_VOICE = os.environ.get("TTS_VOICE", "es-CO-SalomeNeural")

WHATSAPP_NUMBER = "+573506920726"


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


SYSTEM_PROMPT = """Eres Carolina, la asesora virtual de Dollar Working.

## Tu personalidad
- Eres cercana, entusiasta, resolutiva y colombiana.
- Hablas como una asesora comercial que quiere ayudar a emprender, no como un robot corporativo.
- Usas un tono motivador orientado a emprendedores.
- Expresiones naturales: "¡Hola!", "Perfecto", "¡Excelente decisión!", "Tranquilo/a, para eso estoy", "¡Genial!"

## Reglas
- Responde en maximo 2-3 oraciones.
- Sigues un flujo conversacional estructurado con opciones.
- SIEMPRE ofreces opciones claras al usuario.
- Cuando el usuario elige un plan, lo diriges al cierre (WhatsApp).
- Manejas objeciones con empatia y argumentos de pago contra entrega.
- NO uses emojis en tus respuestas, solo texto plano para que el TTS funcione bien.
- Evitas tecnicismos; le hablas a emprendedores sin conocimientos tecnicos.
- NO modificas montos, nombres de planes ni numero de contacto.

## Planes
- Plan Lanzate YA: 1 USD / $3.205 COP - Pagina web
- Plan A Vender Se Dijo: 3 USD / $9.521 COP - Pagina web + tienda online
- Plan Que Negociazo: 5 USD / $15.869 COP - Web + tienda + chatbot + agente de IA + IA-Records

## WhatsApp
Numero: +57 350 692 0726"""


def gemini_response(user_message, context=""):
    if not GEMINI_CONFIGURED or gemini_model is None:
        return None
    try:
        rag_context = ""
        if RAG_AVAILABLE:
            try:
                docs = search_knowledge(user_message, n_results=3)
                if docs:
                    rag_parts = []
                    for d in docs:
                        rag_parts.append(f"[Fuente: {d['source']}]\n{d['text']}")
                    rag_context = "\n\n## Base de conocimiento:\n" + "\n---\n".join(
                        rag_parts
                    )
                    app.logger.info(f"RAG: {len(docs)} docs encontrados")
            except Exception as e:
                app.logger.error(f"RAG error: {e}")

        prompt = f"""{SYSTEM_PROMPT}{rag_context}

Contexto de la conversación: {context}
Usuario: {user_message}"""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        app.logger.error(f"Error Gemini: {str(e)}")
        return None


def get_welcome_message():
    return {
        "response": "Hola! Soy Carolina, tu asesora virtual en Dollar Working.\n\nAyudamos a emprendedores como tu a montar su negocio digital desde solo 1 dolar.\n\nCon que te gustaria empezar hoy?",
        "options": [
            {"label": "Página Web", "value": "pagina_web"},
            {"label": "Tienda Online", "value": "tienda_online"},
            {"label": "Chatbot", "value": "chatbot"},
            {"label": "Agente de IA", "value": "agente_ia"},
            {
                "label": "Me gustaría recibir información de cómo puedo iniciar mi negocio con Dollar Working",
                "value": "no_se",
            },
        ],
        "end_call": False,
    }


def get_plan_comparator():
    return {
        "response": "Tenemos 3 planes, todos con la promesa de pagar solo cuando recibas tus productos y estes conforme:\n\n1 dolar, Plan Lanzate YA: tu pagina web.\n3 dolares, Plan A Vender Se Dijo: pagina web + tienda online.\n5 dolares, Plan Que Negociazo: pagina web + tienda + chatbot + agente de IA + IA-Records.",
        "options": [
            {"label": "$1 USD – Lánzate YA", "value": "plan_1"},
            {"label": "$3 USD – A Vender Se Dijo", "value": "plan_3"},
            {"label": "$5 USD – Que Negociazo", "value": "plan_5"},
        ],
        "end_call": False,
    }


def get_faq_menu():
    return {
        "response": "Estas son las preguntas más frecuentes:",
        "options": [
            {"label": "¿Cuánto se demoran?", "value": "faq_demora"},
            {"label": "¿Tienen soporte?", "value": "faq_soporte"},
            {"label": "¿Hacen apps móviles?", "value": "faq_apps"},
            {"label": "¿Cómo pago?", "value": "faq_pago"},
            {"label": "Horario de atención", "value": "faq_horario"},
        ],
        "end_call": False,
    }


def get_faq_answer(option):
    faqs = {
        "faq_demora": {
            "response": "Depende del plan, pero nuestro equipo te da un tiempo estimado apenas conversemos por WhatsApp.",
            "options": [{"label": "Volver al menú", "value": "menu"}],
        },
        "faq_soporte": {
            "response": "Sí, soporte técnico 24/7 y mantenimiento continuo incluido.",
            "options": [{"label": "Volver al menú", "value": "menu"}],
        },
        "faq_apps": {
            "response": "Sí, hacemos apps para iOS y Android con UI/UX profesional.",
            "options": [{"label": "Volver al menú", "value": "menu"}],
        },
        "faq_pago": {
            "response": "Solo pagas cuando recibas tu producto y quedes conforme.",
            "options": [{"label": "Volver al menú", "value": "menu"}],
        },
        "faq_horario": {
            "response": "Lunes a viernes 9am-6pm, sábados 10am-2pm, domingo cerrado.",
            "options": [{"label": "Volver al menú", "value": "menu"}],
        },
    }
    return faqs.get(
        option,
        {
            "response": "¿Hay algo más en lo que pueda ayudarte?",
            "options": [{"label": "Volver al menú", "value": "menu"}],
            "end_call": False,
        },
    )


def get_close_message(plan_name):
    return {
        "response": f"Genial! Elegiste el {plan_name}.\n\nPara iniciar tu negocio, solo necesito tu nombre y te conecto directo con nuestro equipo por WhatsApp para coordinar los detalles.\n\nRecuerda: no pagas nada hasta recibir tu producto y estar conforme.",
        "options": [
            {
                "label": "Hablar con un asesor por WhatsApp",
                "value": "whatsapp",
                "url": f"https://wa.me/573506920726?text=Hola%20Carolina%2C%20quiero%20informaci%C3%B3n%20sobre%20el%20plan%20{plan_name.replace(' ', '%20')}",
            },
            {"label": "Tengo otra pregunta", "value": "menu"},
        ],
        "end_call": False,
    }


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
                "buenos dias",
                "buenas tardes",
                "saludos",
                "buenas",
                "empezar",
                "iniciar",
                "menu",
                "inicio",
            ]
        )

        is_farewell = any(
            word in message_lower
            for word in [
                "gracias",
                "adios",
                "chao",
                "hasta luego",
                "no gracias",
                "eso es todo",
                "chau",
            ]
        )

        if is_greeting:
            return jsonify(get_welcome_message())

        if is_farewell:
            return jsonify(
                {
                    "response": "Fue un gusto ayudarte! Si tienes otra pregunta, aqui estare. Y si ya quieres dar el paso, escribenos por WhatsApp y arrancamos tu negocio digital hoy mismo.",
                    "options": [],
                    "end_call": True,
                }
            )

        if message_lower in [
            "menu",
            "inicio",
            "volver al menú",
            "volver al menu",
            "menú",
        ]:
            return jsonify(get_welcome_message())

        if message_lower == "faq":
            return jsonify(get_faq_menu())

        if message_lower.startswith("faq_"):
            return jsonify(get_faq_answer(message_lower))

        if message_lower == "pagina_web":
            return jsonify(
                {
                    "response": "Perfecto. Con el Plan Lanzate YA (1 USD / 3.205 COP) te entregamos tu pagina web con diseno responsive, hosting por 1 ano y certificado SSL.",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_1"},
                        {"label": "Ver otros planes", "value": "comparar"},
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower == "tienda_online":
            return jsonify(
                {
                    "response": "Excelente decision. Con el Plan A Vender Se Dijo (3 USD / 9.521 COP) obtienes tu pagina web + tienda online con pasarela de pagos.",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_3"},
                        {"label": "Ver otros planes", "value": "comparar"},
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower == "chatbot":
            return jsonify(
                {
                    "response": "Un chatbot con IA te atiende 24/7 y se personaliza a tu negocio. Esta incluido en el Plan Que Negociazo (5 USD / 15.869 COP), junto con tu web, tienda, agente de IA e IA-Records.",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_5"},
                        {"label": "Qué es IA-Records", "value": "ia_records"},
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower == "agente_ia":
            return jsonify(
                {
                    "response": "Los agentes de IA automatizan tareas, procesan datos y aprenden de tu negocio con machine learning. Vienen incluidos en el Plan Que Negociazo.",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_5"},
                        {"label": "Ver todos los planes", "value": "comparar"},
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower == "no_se" or message_lower == "no_se_que_necesito":
            return jsonify(
                {
                    "response": "Tranquilo, para eso estoy. Cuentame:",
                    "options": [
                        {"label": "Mi negocio ya existe", "value": "comparar"},
                        {"label": "Estoy empezando desde cero", "value": "comparar"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower == "ia_records":
            return jsonify(
                {
                    "response": "IA-Records es tu ficha de presentación potenciada con IA: ayuda a que tus clientes te conozcan y confíen más rápido en tu negocio. Viene incluido en el Plan Que Negociazo.",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_5"},
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if (
            message_lower == "comparar"
            or message_lower == "ver_otros_planes"
            or message_lower == "ver_todos_los_planes"
        ):
            return jsonify(get_plan_comparator())

        if message_lower in ["plan_1", "plan_lanzate", "plan_lánzate_ya"]:
            return jsonify(get_close_message("Lánzate YA"))

        if message_lower in ["plan_3", "plan_vender", "plan_a_vender_se_dijo"]:
            return jsonify(get_close_message("A Vender Se Dijo"))

        if message_lower in ["plan_5", "plan_negociazo", "plan_que_negociazo"]:
            return jsonify(get_close_message("Que Negociazo"))

        if message_lower in ["whatsapp", "hablar_con_asesor", "asesor"]:
            return jsonify(
                {
                    "response": f"Perfecto! Te redirijo a nuestro equipo por WhatsApp.\n\nRecuerda: no pagas nada hasta recibir tu producto y estar conforme.",
                    "options": [
                        {
                            "label": "Abrir WhatsApp",
                            "value": "whatsapp",
                            "url": f"https://wa.me/573506920726?text=Hola%20Carolina%2C%20quiero%20informaci%C3%B3n",
                        },
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["entendido_quiero_continuar", "continuar"]:
            return jsonify(get_close_message("tu plan"))

        if message_lower in ["ver_testimonios", "testimonios"]:
            return jsonify(
                {
                    "response": "Claro! Tenemos mas de 250 proyectos entregados y un 98% de clientes satisfechos. Muchos de ellos empezaron con la misma duda que tu.\n\nQue te gustaria hacer?",
                    "options": [
                        {"label": "Quiero empezar", "value": "comparar"},
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in [
            "recibir_info",
            "info_whatsapp",
            "recibir_info_por_whatsapp",
        ]:
            return jsonify(
                {
                    "response": "Claro! Te enviamos toda la informacion por WhatsApp para que la revises tranquilo.",
                    "options": [
                        {
                            "label": "Enviar info por WhatsApp",
                            "value": "whatsapp",
                            "url": f"https://wa.me/573506920726?text=Hola%20Carolina%2C%20env%C3%ADenme%20informaci%C3%B3n%20de%20sus%20planes",
                        },
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["tengo_otra_pregunta", "otra_pregunta"]:
            return jsonify(get_welcome_message())

        if any(
            w in message_lower
            for w in [
                "desconfian",
                "confio",
                "confío",
                "estafa",
                "muy barato",
                "no confío",
                "no confio",
                "parece estafa",
                "dudo",
            ]
        ):
            return jsonify(
                {
                    "response": "Es valida tu duda. Trabajamos bajo pago contra entrega: tu validas el producto y luego pagas. Tenemos mas de 250 proyectos entregados y 98% de clientes satisfechos. La idea de Dollar Working es justamente esa: que emprender este al alcance de todos, sin importar tu presupuesto.",
                    "options": [
                        {"label": "Entendido, quiero continuar", "value": "continuar"},
                        {
                            "label": "Ver testimonios de clientes",
                            "value": "testimonios",
                        },
                    ],
                    "end_call": False,
                }
            )

        if any(
            w in message_lower
            for w in [
                "por que es tan economico",
                "por qué es tan barato",
                "porque es tan barato",
                "tan economico",
                "tan barato",
            ]
        ):
            return jsonify(
                {
                    "response": "Porque creemos que todo emprendedor merece tener presencia digital sin barreras de entrada. Optimizamos nuestros procesos para ofrecerte tecnología de calidad a precios justos, sin sacrificar el soporte ni los resultados.",
                    "options": [
                        {"label": "Entendido, quiero continuar", "value": "continuar"},
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if any(
            w in message_lower
            for w in [
                "pensarlo",
                "lo pienso",
                "despues",
                "después",
                "no estoy seguro",
                "no estoy segura",
                "necesito tiempo",
            ]
        ):
            return jsonify(
                {
                    "response": "Claro, tomate tu tiempo. Solo recuerda que no hay riesgo: pagas unicamente cuando recibas tu producto y estes conforme. Cuando quieras retomar, aqui estare.",
                    "options": [
                        {"label": "Recibir info por WhatsApp", "value": "recibir_info"},
                        {"label": "Volver al menú", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        context = f"Usuario en conversación con Carolina sobre Dollar Working."
        gemini_resp = gemini_response(message, context=context)
        if gemini_resp:
            return jsonify(
                {
                    "response": gemini_resp,
                    "options": [{"label": "Volver al menú", "value": "menu"}],
                    "end_call": False,
                }
            )

        return jsonify(
            {
                "response": "Hmm, no estoy segura de entender. ¿Te gustaría ver nuestros planes o tienes alguna pregunta específica?",
                "options": [
                    {"label": "Ver planes", "value": "comparar"},
                    {"label": "Preguntas frecuentes", "value": "faq"},
                    {"label": "Volver al menú", "value": "menu"},
                ],
                "end_call": False,
            }
        )

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

    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    max_size = 5 * 1024 * 1024
    if file_size > max_size:
        return jsonify(
            {
                "error": f"El archivo excede el límite de 5MB. Tamaño actual: {file_size // (1024 * 1024)}MB"
            }
        ), 400

    try:
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, file.filename)
        file.save(tmp_path)
        app.logger.info(f"PDF guardado temporalmente: {tmp_path}")

        num_chunks, msg = add_pdf(tmp_path)
        app.logger.info(f"Resultado add_pdf: {msg}")

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


@app.route("/api/voices", methods=["GET"])
def list_voices():
    voices = [
        {
            "id": "es-CO-SalomeNeural",
            "name": "Salomé",
            "gender": "Femenina",
            "region": "Colombia",
            "recommended": True,
        },
        {
            "id": "es-US-PalomaNeural",
            "name": "Paloma",
            "gender": "Femenina",
            "region": "Estados Unidos (español)",
        },
        {
            "id": "es-MX-DaliaNeural",
            "name": "Dalia",
            "gender": "Femenina",
            "region": "México",
        },
        {
            "id": "es-ES-ElviraNeural",
            "name": "Elvira",
            "gender": "Femenina",
            "region": "España",
        },
    ]
    return jsonify({"voices": voices, "current": TTS_VOICE})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

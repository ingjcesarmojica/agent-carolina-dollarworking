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

TTS_VOICE = os.environ.get("TTS_VOICE", "es-MX-DaliaNeural")
TTS_RATE = os.environ.get("TTS_RATE", "+15%")

WHATSAPP_NUMBER = "+573506920726"


async def generate_edge_tts(text, voice=None, rate=None):
    if voice is None:
        voice = TTS_VOICE
    if rate is None:
        rate = TTS_RATE
    communicate = edge_tts.Communicate(text, voice, rate=rate)
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
- Expresiones naturales: Hola, Perfecto, Excelente decision, Tranquilo para eso estoy, Genial

## Reglas
- Responde en maximo dos a tres oraciones.
- Sigues un flujo conversacional estructurado con opciones.
- SIEMPRE ofreces opciones claras al usuario.
- Cuando el usuario elige un plan, lo diriges al cierre por WhatsApp.
- Manejas objeciones con empatia y argumentos de pago contra entrega.
- NO uses emojis en tus respuestas, solo texto plano para que el TTS funcione bien.
- Evitas tecnicismos le hablas a emprendedores sin conocimientos tecnicos.
- NO modificas montos nombres de planes ni numero de contacto.
- NO uses signos de exclamacion ni interrogation.
- NO uses barras diagonales ni slashes.
- NO uses numeros escribe todo en letras por ejemplo uno tres cinco dolares.
- NO uses comas ni puntos y comas al final de las frases.

## Planes
- Plan LanZaTE YA un dolar Pagina web
- Plan A Vender Se Dijo tres dolares Pagina web y tienda online
- Plan Que Negociazo cinco dolares Web tienda chatbot agente de IA e IA Records

## WhatsApp
Numero cincuenta y siete tres cincuenta seiscientos noventa y dos cero siete dos seis"""


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

Contexto de la conversacion: {context}
Usuario: {user_message}"""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        app.logger.error(f"Error Gemini: {str(e)}")
        return None


def is_valid_email(text):
    return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", text))


def is_valid_phone(text):
    cleaned = re.sub(r"[\s\-\(\)\+]", "", text)
    return cleaned.isdigit() and 7 <= len(cleaned) <= 15


def get_welcome_message():
    return {
        "response": "Hola, soy Carolina, tu asesora de Dollar Working, un gusto saludarte, por favor para poder colaborarte con la información, diligencia los siguientes campos:",
        "flow": "ask_name",
        "options": [],
        "end_call": False,
    }


def get_ask_email(name):
    return {
        "response": f"Mucho gusto {name}. Para poder enviarte informacion personalizada cual es tu correo electronico",
        "flow": "ask_email",
        "options": [],
        "end_call": False,
    }


def get_ask_phone(name, email):
    return {
        "response": "Perfecto. Un ultimo dato cual es tu numero de WhatsApp. Asi nuestro equipo puede contactarte directamente",
        "flow": "ask_phone",
        "options": [],
        "end_call": False,
    }


def get_discovery_web(name):
    return {
        "response": "Excelente. Me gustaria saber si ya cuentas con una pagina web de tu negocio",
        "flow": "discovery_web",
        "options": [
            {"label": "Si ya tengo pagina web", "value": "has_web_yes"},
            {"label": "No no tengo", "value": "has_web_no"},
        ],
        "end_call": False,
    }


def get_discovery_store(name, has_web):
    if has_web:
        return {
            "response": "Excelente. Cuanteme esa pagina web tiene tienda en linea",
            "flow": "discovery_store",
            "options": [
                {"label": "Si ya tengo tienda", "value": "has_store_yes"},
                {"label": "No no tengo", "value": "has_store_no"},
            ],
            "end_call": False,
        }
    else:
        return {
            "response": "Perfecto. Cuanteme necesitas una tienda en linea para vender tus productos",
            "flow": "discovery_store",
            "options": [
                {"label": "Si, me interesa", "value": "needs_store_yes"},
                {"label": "No, solo necesito otra cosa", "value": "needs_store_no"},
            ],
            "end_call": False,
        }


def get_discovery_chatbot(name, has_store):
    if has_store:
        return {
            "response": "Muy bien. Esa tienda tiene algun chatbot que responda a tus clientes automaticamente",
            "flow": "discovery_chatbot",
            "options": [
                {"label": "Si ya tengo chatbot", "value": "has_chatbot_yes"},
                {"label": "No no tengo", "value": "has_chatbot_no"},
            ],
            "end_call": False,
        }
    else:
        return {
            "response": "Entendido. Necesitas un chatbot que responda a tus clientes todo el dia",
            "flow": "discovery_chatbot",
            "options": [
                {"label": "Si me interesa", "value": "needs_chatbot_yes"},
                {"label": "No no necesito", "value": "needs_chatbot_no"},
            ],
            "end_call": False,
        }


def get_discovery_agent(name, has_chatbot):
    if has_chatbot:
        return {
            "response": "Genial. Ese chatbot es inteligente o solo responde preguntas basicas. Tiene algun agente de IA que automatice tareas",
            "flow": "discovery_agent",
            "options": [
                {"label": "Si tiene IA completa", "value": "has_agent_yes"},
                {"label": "No es basico", "value": "has_agent_no"},
            ],
            "end_call": False,
        }
    else:
        return {
            "response": "Te gustaria tener un agente de IA que automatice tareas y aprenda de tu negocio",
            "flow": "discovery_agent",
            "options": [
                {"label": "Si me interesa", "value": "needs_agent_yes"},
                {"label": "No no necesito", "value": "needs_agent_no"},
            ],
            "end_call": False,
        }


def get_discovery_ia_records(name, has_agent):
    if has_agent:
        return {
            "response": "Excelente. Ya conoces IA Records. Es una ficha de presentacion potenciada con IA para que tus clientes te conozcan y confien mas rapido",
            "flow": "discovery_ia_records",
            "options": [
                {"label": "Si ya tengo IA Records", "value": "has_ia_records_yes"},
                {"label": "No no lo conozco", "value": "has_ia_records_no"},
            ],
            "end_call": False,
        }
    else:
        return {
            "response": "Te gustaria tener IA Records. Es una ficha de presentacion potenciada con IA para que tus clientes te conozcan y confien mas rapido",
            "flow": "discovery_ia_records",
            "options": [
                {"label": "Si me interesa", "value": "needs_ia_records_yes"},
                {"label": "No no necesito", "value": "needs_ia_records_no"},
            ],
            "end_call": False,
        }


def get_discovery_summary(
    name, has_web, has_store, has_chatbot, has_agent, has_ia_records
):
    services_has = []
    services_needs = []

    if has_web:
        services_has.append("Pagina Web")
    else:
        services_needs.append("Pagina Web")

    if has_store:
        services_has.append("Tienda Online")
    else:
        services_needs.append("Tienda Online")

    if has_chatbot:
        services_has.append("Chatbot con IA")
    else:
        services_needs.append("Chatbot con IA")

    if has_agent:
        services_needs.append("Agente de IA mejorado")
    else:
        services_needs.append("Agente de IA")

    if has_ia_records:
        services_has.append("IA-Records")
    else:
        services_needs.append("IA-Records")

    response_parts = []
    if services_has:
        response_parts.append(f"Lo que ya tienes {', '.join(services_has)}")
    if services_needs:
        response_parts.append(
            f"Lo que te podemos mejorar o agregar {', '.join(services_needs)}"
        )

    if len(services_needs) == 5:
        plan = "Que Negociazo"
        price = "cinco dolares"
        desc = "nuestro paquete completo con todo web tienda chatbot con IA agente de IA e IA Records"
    elif "Pagina Web" in services_needs and "Tienda Online" in services_needs:
        plan = "A Vender Se Dijo"
        price = "tres dolares"
        desc = "tu pagina web y tienda online con pasarela de pagos"
    elif "Pagina Web" in services_needs:
        plan = "LanZaTE YA"
        price = "un dolar"
        desc = "tu pagina web profesional con diseno responsive"
    else:
        plan = "Que Negociazo"
        price = "cinco dolares"
        desc = "mejorar y complementar lo que ya tienes con tecnologia de punta"

    summary_text = "\n".join(response_parts)

    return {
        "response": f"Esto es lo que encontre\n\n{summary_text}\n\nTe recomiendo el Plan {plan} {price}. Con este plan obtienes {desc}.\n\nLo mejor no pagas nada hasta recibir tu producto y estar conforme.\n\nQuieres avanzar con este plan",
        "flow": "confirm_plan",
        "plan": plan,
        "options": [
            {"label": "Si quiero este plan", "value": "advance"},
            {"label": "Ver otro plan", "value": "comparar"},
            {"label": "Tengo dudas", "value": "have_doubts"},
        ],
        "end_call": False,
    }


def get_plan_comparator():
    return {
        "response": "Tenemos tres planes increibles todos con la promesa de pagar solo cuando recibas tus productos y estes conforme\n\nUn dolar Plan LanZaTE YA tu pagina web profesional\nTres dolares Plan A Vender Se Dijo pagina web y tienda online\nCinco dolares Plan Que Negociazo web tienda chatbot agente de IA e IA Records\n\nCual se ajusta mas a lo que necesitas",
        "flow": "choose_plan",
        "options": [
            {"label": "Un dolar LanZaTE YA", "value": "plan_1"},
            {"label": "Tres dolares A Vender Se Dijo", "value": "plan_3"},
            {"label": "Cinco dolares Que Negociazo", "value": "plan_5"},
            {"label": "Necesito mas informacion", "value": "need_more_info"},
        ],
        "end_call": False,
    }


def get_recommend_plan(interest, name):
    recommendations = {
        "interest_pagina_web": {
            "plan": "LanZaTE YA",
            "price": "un dolar",
            "desc": "una pagina web profesional con diseno responsive hosting por un ano y certificado SSL",
        },
        "interest_tienda_online": {
            "plan": "A Vender Se Dijo",
            "price": "tres dolares",
            "desc": "tu pagina web y tienda online con pasarela de pagos para vender desde el dia uno",
        },
        "interest_chatbot": {
            "plan": "Que Negociazo",
            "price": "cinco dolares",
            "desc": "todo incluido web tienda chatbot con IA agente de IA e IA Records",
        },
        "interest_agente_ia": {
            "plan": "Que Negociazo",
            "price": "cinco dolares",
            "desc": "todo incluido web tienda chatbot con IA agente de IA e IA Records",
        },
        "interest_todos": {
            "plan": "Que Negociazo",
            "price": "cinco dolares",
            "desc": "nuestro paquete completo con todos los servicios web tienda chatbot agente de IA e IA Records",
        },
        "interest_no_se": {
            "plan": "Que Negociazo",
            "price": "cinco dolares",
            "desc": "todos nuestros servicios incluidos",
        },
    }

    rec = recommendations.get(interest, recommendations["interest_todos"])

    return {
        "response": f"Basado en lo que me cuentas te recomiendo el Plan {rec['plan']} {rec['price']}.\n\nCon este plan obtienes {rec['desc']}.\n\nLo mejor no pagas nada hasta recibir tu producto y estar conforme sin riesgos.\n\nQuieres avanzar con este plan",
        "flow": "confirm_plan",
        "plan": rec["plan"],
        "options": [
            {"label": "Si quiero este plan", "value": f"confirm_{interest}"},
            {"label": "Ver otro plan", "value": "comparar"},
            {"label": "Tengo dudas", "value": "have_doubts"},
        ],
        "end_call": False,
    }


def get_close_message(plan_name, name):
    return {
        "response": f"Excelente decision. El Plan {plan_name} es perfecto para arrancar tu negocio digital.\n\nNuestro equipo se pondra en contacto contigo por WhatsApp para coordinar los detalles y comenzar tu proyecto.\n\nRecuerda no pagas nada hasta recibir tu producto y estar conforme",
        "flow": "conversion",
        "options": [
            {
                "label": "Hablar con un asesor por WhatsApp",
                "value": "whatsapp",
                "url": f"https://wa.me/573506920726?text=Hola%20Carolina%2C%20soy%20{name.replace(' ', '%20')}%2C%20quiero%20el%20plan%20{plan_name.replace(' ', '%20')}",
            },
            {"label": "Tengo otra pregunta", "value": "menu"},
        ],
        "end_call": False,
    }


def get_faq_menu():
    return {
        "response": "Estas son las preguntas mas frecuentes de nuestros clientes",
        "flow": "faq",
        "options": [
            {"label": "Cuanto se demoran", "value": "faq_demora"},
            {"label": "Tienen soporte", "value": "faq_soporte"},
            {"label": "Hacen apps moviles", "value": "faq_apps"},
            {"label": "Como pago", "value": "faq_pago"},
            {"label": "Horario de atencion", "value": "faq_horario"},
        ],
        "end_call": False,
    }


def get_faq_answer(option):
    faqs = {
        "faq_demora": {
            "response": "Depende del plan pero nuestro equipo te da un tiempo estimado apenas conversemos por WhatsApp",
            "flow": "faq",
            "options": [{"label": "Volver al menu", "value": "menu"}],
        },
        "faq_soporte": {
            "response": "Si soporte tecnico todo el dia y mantenimiento continuo incluido en todos los planes",
            "flow": "faq",
            "options": [{"label": "Volver al menu", "value": "menu"}],
        },
        "faq_apps": {
            "response": "Si hacemos apps para iOS y Android con interfaz profesional. Incluido en el Plan Que Negociazo",
            "flow": "faq",
            "options": [{"label": "Volver al menu", "value": "menu"}],
        },
        "faq_pago": {
            "response": "Solo pagas cuando recibas tu producto y quedes conforme. Sin anticipos sin sorpresas",
            "flow": "faq",
            "options": [{"label": "Volver al menu", "value": "menu"}],
        },
        "faq_horario": {
            "response": "Lunes a viernes de nueve a seis de la manana sabados de diez a dos del mediodia domingo cerrado",
            "flow": "faq",
            "options": [{"label": "Volver al menu", "value": "menu"}],
        },
    }
    return faqs.get(
        option,
        {
            "response": "Hay algo mas en lo que pueda ayudarte",
            "flow": "faq",
            "options": [{"label": "Volver al menu", "value": "menu"}],
            "end_call": False,
        },
    )


def get_farewell(name):
    return {
        "response": "Fue un gusto ayudarte. Si tienes otra pregunta aqui estare. Y si ya quieres dar el paso escribenos por WhatsApp y arrancamos tu negocio digital hoy mismo",
        "flow": "end",
        "options": [],
        "end_call": True,
    }


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message", "")
        flow = data.get("flow", "")
        user_data = data.get("user_data", {})

        if not message:
            return jsonify({"error": "No message provided"}), 400

        message_lower = message.lower().strip()
        name = user_data.get("name", "")

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

        if is_greeting and not flow:
            return jsonify(get_welcome_message())

        if is_farewell:
            return jsonify(get_farewell(name))

        if message_lower in [
            "menu",
            "inicio",
            "volver al menu",
        ]:
            return jsonify(get_welcome_message())

        if message_lower == "faq" or flow == "faq":
            if message_lower.startswith("faq_"):
                return jsonify(get_faq_answer(message_lower))
            return jsonify(get_faq_menu())

        if message_lower == "lead_form_submitted" and flow == "ask_name":
            return jsonify(get_discovery_web(name))

        if message_lower == "continuar_llamada":
            if flow == "" or flow == "ask_name":
                return jsonify(
                    {
                        "response": f"Hola {name} soy Carolina. Continuamos con la conversacion. En que puedo ayudarte",
                        "flow": flow,
                        "options": [],
                        "end_call": False,
                    }
                )
            else:
                return jsonify(
                    {
                        "response": f"Hola {name} retomamos donde quedamos. En que puedo ayudarte",
                        "flow": flow,
                        "options": [],
                        "end_call": False,
                    }
                )

        if flow == "ask_name":
            cleaned_name = message.strip().title()
            if len(cleaned_name) < 2:
                return jsonify(
                    {
                        "response": "Por favor, ingresa tu nombre completo.",
                        "flow": "ask_name",
                        "options": [],
                        "end_call": False,
                    }
                )
            return jsonify(get_ask_email(cleaned_name))

        if flow == "ask_email":
            email = message.strip().lower()
            if not is_valid_email(email):
                return jsonify(
                    {
                        "response": "El correo no parece valido. Por favor, ingresa tu correo electronico (ejemplo: tu@email.com).",
                        "flow": "ask_email",
                        "options": [],
                        "end_call": False,
                    }
                )
            return jsonify(get_ask_phone(name, email))

        if flow == "ask_phone":
            phone = message.strip()
            if not is_valid_phone(phone):
                return jsonify(
                    {
                        "response": "El numero no parece valido. Por favor, ingresa tu numero de WhatsApp con codigo de pais (ejemplo: +573001234567).",
                        "flow": "ask_phone",
                        "options": [],
                        "end_call": False,
                    }
                )
            return jsonify(get_discovery_web(name))

        if flow == "discovery_web":
            has_web = message_lower == "has_web_yes"
            user_data["has_web"] = has_web
            return jsonify(get_discovery_store(name, has_web))

        if flow == "discovery_store":
            has_store = (
                message_lower == "has_store_yes" or message_lower == "needs_store_yes"
            )
            user_data["has_store"] = has_store
            return jsonify(get_discovery_chatbot(name, has_store))

        if flow == "discovery_chatbot":
            has_chatbot = (
                message_lower == "has_chatbot_yes"
                or message_lower == "needs_chatbot_yes"
            )
            user_data["has_chatbot"] = has_chatbot
            return jsonify(get_discovery_agent(name, has_chatbot))

        if flow == "discovery_agent":
            has_agent = (
                message_lower == "has_agent_yes" or message_lower == "needs_agent_yes"
            )
            user_data["has_agent"] = has_agent
            return jsonify(get_discovery_ia_records(name, has_agent))

        if flow == "discovery_ia_records":
            has_ia_records = (
                message_lower == "has_ia_records_yes"
                or message_lower == "needs_ia_records_yes"
            )
            user_data["has_ia_records"] = has_ia_records
            return jsonify(
                get_discovery_summary(
                    name,
                    user_data.get("has_web", False),
                    user_data.get("has_store", False),
                    user_data.get("has_chatbot", False),
                    user_data.get("has_agent", False),
                    has_ia_records,
                )
            )

        if flow == "service_interest":
            return jsonify(get_recommend_plan(message_lower, name))

        if flow == "choose_plan":
            if message_lower == "need_more_info":
                return jsonify(
                    {
                        "response": "Todos nuestros planes incluyen hosting SSL soporte todo el dia y pago contra entrega.\n\nElige el que mas se ajuste a tu negocio",
                        "flow": "choose_plan",
                        "options": [
                            {"label": "Un dolar LanZaTE YA", "value": "plan_1"},
                            {
                                "label": "Tres dolares A Vender Se Dijo",
                                "value": "plan_3",
                            },
                            {"label": "Cinco dolares Que Negociazo", "value": "plan_5"},
                        ],
                        "end_call": False,
                    }
                )

        if flow == "confirm_plan":
            if message_lower == "have_doubts":
                return jsonify(
                    {
                        "response": "Tranquilo es normal tener dudas. Recuerda que no pagas nada hasta recibir tu producto y estar conforme. No hay riesgo.\n\nQue duda tienes",
                        "flow": "confirm_plan",
                        "options": [
                            {"label": "Es muy barato desconfio", "value": "distrust"},
                            {"label": "Necesito pensarlo", "value": "think_about"},
                            {"label": "Quiero ver otros planes", "value": "comparar"},
                            {"label": "Quiero avanzar", "value": "advance"},
                        ],
                        "end_call": False,
                    }
                )

        if message_lower == "advance" or message_lower.startswith("confirm_"):
            plan_map = {
                "confirm_interest_pagina_web": "LanZaTE YA",
                "confirm_interest_tienda_online": "A Vender Se Dijo",
                "confirm_interest_chatbot": "Que Negociazo",
                "confirm_interest_agente_ia": "Que Negociazo",
                "confirm_interest_todos": "Que Negociazo",
                "confirm_interest_no_se": "Que Negociazo",
            }
            plan_name = plan_map.get(message_lower, "tu plan")
            return jsonify(get_close_message(plan_name, name))

        if message_lower in [
            "distrust",
            "dudo",
            "estafa",
            "no confio",
            "parece estafa",
        ]:
            return jsonify(
                {
                    "response": "Es valida tu duda. Trabajamos bajo pago contra entrega tu validas el producto y luego pagas. Tenemos mas de doscientos cincuenta proyectos entregados y noventa y ocho por ciento de clientes satisfechos. Dollar Working existe para que emprender este al alcance de todos",
                    "flow": "confirm_plan",
                    "options": [
                        {"label": "Entendido quiero avanzar", "value": "advance"},
                        {"label": "Ver testimonios", "value": "testimonios"},
                        {"label": "Tengo otra duda", "value": "have_doubts"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in [
            "think_about",
            "pensarlo",
            "lo pienso",
            "despues",
            "necesito tiempo",
        ]:
            return jsonify(
                {
                    "response": "No hay presion. Solo recuerda que no hay riesgo pagas unicamente cuando recibas tu producto y estes conforme. Cuando quieras retomar aqui estare",
                    "flow": "confirm_plan",
                    "options": [
                        {"label": "Recibir info por WhatsApp", "value": "recibir_info"},
                        {"label": "Quiero avanzar ahora", "value": "advance"},
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower == "testimonios":
            return jsonify(
                {
                    "response": "Tenemos mas de doscientos cincuenta proyectos entregados y un noventa y ocho por ciento de clientes satisfechos. Muchos empezaron con la misma duda que tu.\n\nQue te gustaria hacer",
                    "flow": "confirm_plan",
                    "options": [
                        {"label": "Quiero avanzar", "value": "advance"},
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["recibir_info", "info_whatsapp"]:
            return jsonify(
                {
                    "response": "Claro. Te enviamos toda la informacion por WhatsApp para que la revises tranquilo",
                    "flow": "conversion",
                    "options": [
                        {
                            "label": "Enviar info por WhatsApp",
                            "value": "whatsapp",
                            "url": f"https://wa.me/573506920726?text=Hola%20Carolina%2C%20soy%20{name.replace(' ', '%20')}%2C%20envienme%20informacion%20de%20sus%20planes",
                        },
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["whatsapp", "hablar_con_asesor", "asesor"]:
            return jsonify(
                {
                    "response": f"Perfecto. Te redirijo a nuestro equipo por WhatsApp.\n\nRecuerda no pagas nada hasta recibir tu producto y estar conforme",
                    "flow": "conversion",
                    "options": [
                        {
                            "label": "Abrir WhatsApp",
                            "value": "whatsapp",
                            "url": f"https://wa.me/573506920726?text=Hola%20Carolina%2C%20soy%20{name.replace(' ', '%20')}%2C%20quiero%20informacion",
                        },
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["pagina_web"]:
            return jsonify(
                {
                    "response": "Con el Plan LanZaTE YA un dolar te entregamos tu pagina web con diseno responsive hosting por un ano y certificado SSL",
                    "flow": "choose_plan",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_1"},
                        {"label": "Ver otros planes", "value": "comparar"},
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["tienda_online"]:
            return jsonify(
                {
                    "response": "Excelente decision. Con el Plan A Vender Se Dijo tres dolares obtienes tu pagina web y tienda online con pasarela de pagos",
                    "flow": "choose_plan",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_3"},
                        {"label": "Ver otros planes", "value": "comparar"},
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["chatbot"]:
            return jsonify(
                {
                    "response": "Un chatbot con IA te atiende todo el dia. Esta incluido en el Plan Que Negociazo cinco dolares junto con tu web tienda agente de IA e IA Records",
                    "flow": "choose_plan",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_5"},
                        {"label": "Que es IA Records", "value": "ia_records"},
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["agente_ia"]:
            return jsonify(
                {
                    "response": "Los agentes de IA automatizan tareas procesan datos y aprenden de tu negocio. Vienen incluidos en el Plan Que Negociazo",
                    "flow": "choose_plan",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_5"},
                        {"label": "Ver todos los planes", "value": "comparar"},
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["ia_records"]:
            return jsonify(
                {
                    "response": "IA Records es tu ficha de presentacion potenciada con IA. Ayuda a que tus clientes te conozcan y confien mas rapido. Incluido en el Plan Que Negociazo",
                    "flow": "choose_plan",
                    "options": [
                        {"label": "Quiero este plan", "value": "plan_5"},
                        {"label": "Volver al menu", "value": "menu"},
                    ],
                    "end_call": False,
                }
            )

        if message_lower in ["comparar", "ver_otros_planes", "ver_todos_los_planes"]:
            return jsonify(get_plan_comparator())

        if message_lower in ["plan_1", "plan_lanzate", "plan_lanzate_ya"]:
            return jsonify(get_close_message("LanZaTE YA", name))

        if message_lower in ["plan_3", "plan_vender", "plan_a_vender_se_dijo"]:
            return jsonify(get_close_message("A Vender Se Dijo", name))

        if message_lower in ["plan_5", "plan_negociazo", "plan_que_negociazo"]:
            return jsonify(get_close_message("Que Negociazo", name))

        if message_lower in ["tengo_otra_pregunta", "otra_pregunta"]:
            return jsonify(get_welcome_message())

        context = f"Usuario en conversacion con Carolina sobre Dollar Working. Datos: nombre={name}."
        gemini_resp = gemini_response(message, context=context)
        if gemini_resp:
            return jsonify(
                {
                    "response": gemini_resp,
                    "flow": flow,
                    "options": [{"label": "Volver al menu", "value": "menu"}],
                    "end_call": False,
                }
            )

        return jsonify(
            {
                "response": "No estoy segura de entender. Que te gustaria hacer",
                "flow": flow,
                "options": [
                    {"label": "Ver planes", "value": "comparar"},
                    {"label": "Preguntas frecuentes", "value": "faq"},
                    {"label": "Volver al menu", "value": "menu"},
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

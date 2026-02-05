import streamlit as st
import os
from openai import OpenAI
from pinecone import Pinecone
import anthropic

# --- 1. CONFIGURATION ---
# This block allows the app to work both Locally (if you set env vars) and on Streamlit Cloud (Secrets)
try:
    # Try loading from Streamlit Cloud Secrets first
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except (FileNotFoundError, KeyError):
    # If running locally without secrets.toml, you can set keys here manually or via env vars
    # (If you already set them in your previous step, you can keep them hardcoded here for local testing)
    pass 
    # Note: If you have hardcoded keys, paste them back here if the secrets fail locally.
    # Example: os.environ["OPENAI_API_KEY"] = "sk-..."

INDEX_HOST = "veraibot1536-o0tqsfu.svc.aped-4627-b74a.pinecone.io"
NAMESPACES = ["book-mybook-cs", "blog-cs", "podcast_cs"]

# --- 2. SETUP PAGE (CZECH UI) ---
st.set_page_config(page_title="AI Kouč Věra Svach", page_icon="🌱") # Browser Tab Name
st.title("🌱 AI Kouč (Věra Svach)") # Main Title
st.markdown("Zeptejte se na cokoliv ohledně seberozvoje, stresu nebo mindfulness.") # Subtitle

# --- 3. INITIALIZE CLIENTS ---
@st.cache_resource
def init_clients():
    # If keys are missing, stop the app nicely
    if "OPENAI_API_KEY" not in os.environ:
        st.error("Chybí API klíče. (Missing API Keys)")
        st.stop()
    return OpenAI(), anthropic.Anthropic(), Pinecone(api_key=PINECONE_API_KEY)

openai_client, anthropic_client, pc = init_clients()
index = pc.Index(host=INDEX_HOST)

def get_embedding(text):
    text = text.replace("\n", " ")
    response = openai_client.embeddings.create(
        input=[text], model="text-embedding-3-small"
    )
    return response.data[0].embedding

def retrieve_context(query):
    try:
        query_vector = get_embedding(query)
    except Exception:
        return "", ""

    all_matches = []
    for ns in NAMESPACES:
        try:
            results = index.query(
                namespace=ns, vector=query_vector, top_k=3, include_metadata=True
            )
            for match in results['matches']:
                match['source_namespace'] = ns
                all_matches.append(match)
        except Exception:
            pass

    sorted_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)
    
    contexts = []
    debug_text = "" 
    
    for match in sorted_matches[:5]:
        text_content = match['metadata'].get('text', '')
        source = match.get('source_namespace', 'unknown')
        score = match.get('score', 0)
        
        if text_content:
            contexts.append(text_content)
            # Translate "Source" and "Relevance" for the proof section
            debug_text += f"--- [Zdroj: {source} | Relevence: {score:.2f}] ---\n{text_content[:300]}...\n\n"
            
    return "\n\n".join(contexts), debug_text

def get_response(user_input):
    context, debug_text = retrieve_context(user_input)
    
    if not context:
        return "Omlouvám se, ale v databázi jsem nenašla žádné relevantní informace.", ""
        
    system_prompt = f"""
    You are an AI Coach modeled after Vera Svach.
    
    CRITICAL RULE:
    Answer based ONLY on the context below. If the answer is not in the context, say "Based on the available content, I don't know."
    
    CONTEXT:
    {context}
    
    TONE:
    Be empathetic, concise, and professional.
    LANGUAGE: The user asks in Czech, you MUST answer in Czech.
    """
    
    models_to_try = ["claude-3-5-sonnet-latest", "claude-3-opus-latest", "claude-3-haiku-20240307"]
    
    for model in models_to_try:
        try:
            msg = anthropic_client.messages.create(
                model=model, max_tokens=1000, temperature=0.3, system=system_prompt,
                messages=[{"role": "user", "content": user_input}]
            )
            return msg.content[0].text, debug_text
        except:
            continue
            
    return "Chyba: Nepodařilo se připojit k AI.", ""

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
             # "View sources in history"
             with st.expander("🔍 Zobrazit zdroje (Historie)"):
                st.text(message["sources"])

# Input box placeholder text translated
if prompt := st.chat_input("Napište svou otázku..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # "Searching and Thinking..." translated
        with st.spinner("Hledám v databázi a přemýšlím..."):
            response_text, sources_text = get_response(prompt)
            st.markdown(response_text)
            
            # "View Retrieved Context" translated
            with st.expander("🔍 Zobrazit použité texty (Důkaz)"):
                st.text(sources_text)
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources_text
    })

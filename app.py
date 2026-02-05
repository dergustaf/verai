import streamlit as st
import os
from openai import OpenAI
from pinecone import Pinecone
import anthropic

# --- 1. CONFIGURATION ---
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except (FileNotFoundError, KeyError):
    pass

INDEX_HOST = "veraibot1536-o0tqsfu.svc.aped-4627-b74a.pinecone.io"
NAMESPACES = ["book-mybook-cs", "blog-cs", "podcast_cs"]

# --- 2. SETUP PAGE ---
st.set_page_config(page_title="AI Kouč Věra Svach", page_icon="🌱")
st.title("🌱 AI Kouč (Věra Svach)")
st.markdown("Zeptejte se na cokoliv ohledně seberozvoje, stresu nebo mindfulness.")

# --- 3. INITIALIZE CLIENTS ---
@st.cache_resource
def init_clients():
    if "OPENAI_API_KEY" not in os.environ:
        st.error("Chybí API klíče.")
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
            debug_text += f"--- [Zdroj: {source} | Relevence: {score:.2f}] ---\n{text_content[:300]}...\n\n"
            
    return "\n\n".join(contexts), debug_text

def get_response(user_input, chat_history):
    # 1. Get Context based on the CURRENT question
    context, debug_text = retrieve_context(user_input)
    
    if not context:
        context_message = "V databázi jsem nenašla přímou odpověď, zkusím odpovědět obecněji."
    else:
        context_message = context

    # 2. Build the System Prompt
    system_prompt = f"""
    Jsi AI kouč založený na filozofii Věry Svach.
    
    KONTEXT Z DATABÁZE (pro aktuální otázku):
    {context_message}
    
    INSTRUKCE:
    1. Primárně vycházej z kontextu výše.
    2. Pokud uživatel navazuje na předchozí konverzaci (např. používá "ona", "to"), použij historii chatu k pochopení souvislostí.
    3. Buď empatická, stručná a mluv česky.
    """
    
    # 3. PREPARE FULL CONVERSATION HISTORY
    # We create a list of messages starting with history, then adding the new query
    api_messages = []
    
    # Add history (excluding the very last one which is the current prompt we just added in UI)
    for msg in chat_history[:-1]: 
        # Anthropic only accepts 'role' and 'content', so we filter out 'sources'
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current user question
    api_messages.append({"role": "user", "content": user_input})
    
    models_to_try = ["claude-3-5-sonnet-latest", "claude-3-opus-latest", "claude-3-haiku-20240307"]
    
    for model in models_to_try:
        try:
            msg = anthropic_client.messages.create(
                model=model, max_tokens=1000, temperature=0.3, system=system_prompt,
                messages=api_messages # <--- WE NOW SEND THE FULL HISTORY
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
             with st.expander("🔍 Zobrazit zdroje (Historie)"):
                st.text(message["sources"])

if prompt := st.chat_input("Napište svou otázku..."):
    # Append user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Hledám v databázi a přemýšlím..."):
            # Pass the prompt AND the full session state messages
            response_text, sources_text = get_response(prompt, st.session_state.messages)
            st.markdown(response_text)
            
            with st.expander("🔍 Zobrazit použité texty (Důkaz)"):
                st.text(sources_text)
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources_text
    })

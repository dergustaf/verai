import streamlit as st
import os
from openai import OpenAI
from pinecone import Pinecone
import anthropic

# --- 1. CONFIGURATION ---
# We use st.secrets to load keys safely from the cloud setting
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except FileNotFoundError:
    st.error("Secrets not found. Please set up your API keys in the dashboard.")
    st.stop()
    
INDEX_HOST = "veraibot1536-o0tqsfu.svc.aped-4627-b74a.pinecone.io"
NAMESPACES = ["book-mybook-cs", "blog-cs", "podcast_cs"]

# --- 2. SETUP PAGE ---
st.set_page_config(page_title="Vera Svach AI Coach", page_icon="🌱")
st.title("🌱 AI Coach (Vera Svach)")
st.markdown("Ask anything about self-development, stress, or mindfulness.")

# --- 3. INITIALIZE CLIENTS ---
@st.cache_resource
def init_clients():
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
    """
    Returns the retrieved text chunks to prove where the answer came from.
    """
    try:
        query_vector = get_embedding(query)
    except Exception:
        return ""

    all_matches = []
    # Search all namespaces
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

    # Sort by relevance and take top 5
    sorted_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)
    
    contexts = []
    # We create a formatted string showing the Score and Source for each chunk
    debug_text = "" 
    
    for match in sorted_matches[:5]:
        text_content = match['metadata'].get('text', '')
        source = match.get('source_namespace', 'unknown')
        score = match.get('score', 0)
        
        if text_content:
            contexts.append(text_content)
            # Create a nice label for the UI
            debug_text += f"--- [Source: {source} | Relevance: {score:.2f}] ---\n{text_content[:300]}...\n\n"
            
    return "\n\n".join(contexts), debug_text

def get_response(user_input):
    # 1. Get Context AND the "Proof" (debug_text)
    context, debug_text = retrieve_context(user_input)
    
    if not context:
        return "I am sorry, I couldn't find any relevant information in the database.", ""
        
    system_prompt = f"""
    You are an AI Coach modeled after Vera Svach.
    
    CRITICAL RULE:
    Answer based ONLY on the context below. If the answer is not in the context, say "I don't know based on the provided articles."
    
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
            # Return both the answer AND the debug source text
            return msg.content[0].text, debug_text
        except:
            continue
            
    return "Error: Could not connect to Anthropic AI.", ""

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are stored sources in history, show them too (optional, but good for review)
        if "sources" in message:
             with st.expander("🔍 View Retrieved Context (History)"):
                st.text(message["sources"])

# User Input
if prompt := st.chat_input("Type your question here..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching database & Thinking..."):
            response_text, sources_text = get_response(prompt)
            
            st.markdown(response_text)
            
            # THE MAGIC PART: Show the sources
            with st.expander("🔍 View Retrieved Context (Proof)"):
                st.text(sources_text)
            
    # Save answer to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources_text # Save sources so they persist if you scroll up
    })

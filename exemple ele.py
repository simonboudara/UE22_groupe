import epo
import chat_gpt2 as chat_gpt
import transfer_learning_str as st
st.title("Chatbot conversation")

# Initialize session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'reference_number' not in st.session_state:
    st.session_state['reference_number'] = ""
if 'full_discussion' not in st.session_state:
    st.session_state['full_discussion'] = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
if 'show_text_input' not in st.session_state:
    st.session_state['show_text_input'] = True

# Function to display a chat bubble
def display_bubble(text, is_user=True):
    bubble_class = "user-bubble" if is_user else "bot-bubble"
    st.markdown(f"""
        <div class="{bubble_class}">
            {text}
        </div>
        """, unsafe_allow_html=True)

def process_text(user_question):
    # Exemple de traitement du texte (à remplacer par votre logique)
    prompt = f"Retourne uniquement numero de publication du document mentionné dans cette question : {user_question}. Ne dit rien d'autre que cet identifiant. Si il n'y a pas de numéro de publication, écris 'None'."
    
    answer = chat_gpt.ask_reference_number(prompt)
    
    if answer == 'None':
        prompt = f"\nRéponds à la question suivante au sujet du document de brevet précédent : {user_question}"
    else:
        st.session_state['reference_number'] = answer
        reference_number = st.session_state['reference_number']
        
        access_token = epo.ops_authentification()
        epo_document = epo.ops_get_fulltext(access_token, reference_number)
        prompt = f"\nVoici le texte complet du document de brevet {reference_number} : {epo_document}, \n\nRéponds à la question suivante au sujet de ce document de brevet : {user_question}"

    st.session_state['full_discussion'].append({"role": "user", "content": prompt})
    response_content = chat_gpt.ask(st.session_state['full_discussion'])
    st.session_state['full_discussion'].append({"role": "system", "content": response_content})

    return response_content

# Function to handle message sending
def send_message():
    if st.session_state.user_input:
        response = "Réponse du bot : " + process_text(st.session_state.user_input)
        st.session_state['messages'].append({"message": response, "is_user": False})
        st.session_state['messages'].append({"message": st.session_state.user_input, "is_user": True})
        st.session_state['user_input'] = ""  # Clear the input after processing
        st.session_state['show_text_input'] = False  # Hide text input temporarily

# Function to reset the text input visibility
def reset_input_visibility():
    st.session_state['show_text_input'] = True

def reset():
    st.session_state['show_text_input'] = True
    st.session_state['messages'] = []
    st.session_state['reference_number'] = ""


# CSS styles for chat bubbles
st.markdown("""
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        width: fit-content;
        max-width: 80%;
        text-align: left;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #EAEAEA;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        width: fit-content;
        max-width: 80%;
        text-align: left;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column-reverse;
        align-items: flex-start;
    }
    .user-container {
        align-items: flex-end;
    }
    </style>
    """, unsafe_allow_html=True)

# Display the text input conditionally
if st.session_state['show_text_input']:
    user_input = st.text_input("Enter your question : ", key="user_input")
    st.button("Send", on_click=send_message)
else:
    col1, col2 = st.columns([2, 8])
    with col1:
        st.button("New message", on_click=reset_input_visibility)
    with col2:
        st.button("Reset", on_click=reset)


# Display the chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in reversed(st.session_state['messages']):
    bubble_class = "user-container" if msg['is_user'] else ""
    st.markdown(f'<div class="{bubble_class}">', unsafe_allow_html=True)
    display_bubble(msg['message'], is_user=msg['is_user'])
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
import streamlit as st
import nltk
import networkx as nx
import matplotlib.pyplot as plt

# Download tokenizer
nltk.download('punkt', quiet=True)

st.set_page_config(page_title="Semantic Disambiguation Engine", layout="wide")
st.title("🧠 Real-Time Semantic Disambiguation Engine")
st.markdown("### Explore how semantic interpretation resolves ambiguity in language")

# --- Input Section ---
text = st.text_input("Enter a sentence:", "I saw the man with the telescope")

# --- Lexical Knowledge Base ---
lexical_db = {
    "bank": ["financial institution", "river side"],
    "bat": ["flying mammal", "cricket equipment"],
    "telescope": ["optical instrument"],
    "man": ["human male", "person"]
}

# --- Rule-based Lexical Disambiguation ---
def lexical_disambiguation(sentence):
    words = nltk.word_tokenize(sentence.lower())
    senses = {}
    for w in words:
        if w in lexical_db:
            if "river" in sentence or "water" in sentence:
                senses[w] = lexical_db[w][1]
            elif "money" in sentence or "cricket" in sentence:
                senses[w] = lexical_db[w][0]
            else:
                senses[w] = lexical_db[w][0]
    return senses

# --- Structural Disambiguation Example ---
def structural_disambiguation(sentence):
    if "with" in sentence:
        return "⚠️ Possible PP-attachment ambiguity detected ('with' phrase)."
    return "✅ No major structural ambiguity detected."

# --- Semantic Frame Extraction ---
def semantic_frame(sentence):
    tokens = nltk.word_tokenize(sentence)
    frame = {"Actor": None, "Action": None, "Object": None, "Modifier": None}
    pos_tags = nltk.pos_tag(tokens)
    for (word, tag) in pos_tags:
        if tag.startswith("VB"):
            frame["Action"] = word
        elif tag.startswith("NN") and not frame["Actor"]:
            frame["Actor"] = word
        elif tag.startswith("NN"):
            frame["Object"] = word
        elif tag.startswith("IN"):
            frame["Modifier"] = word
    return frame

# --- UNL-like Representation ---
def to_unl(frame):
    unl = []
    if frame["Action"]:
        unl.append(f"agt({frame['Action']},{frame['Actor']})")
        unl.append(f"obj({frame['Action']},{frame['Object']})")
        if frame["Modifier"]:
            unl.append(f"mod({frame['Action']},{frame['Modifier']})")
    return unl

# --- Display Results ---
if text:
    st.subheader("1️⃣ Lexical Disambiguation")
    senses = lexical_disambiguation(text)
    if senses:
        for k, v in senses.items():
            st.write(f"**{k}** → {v}")
    else:
        st.write("No lexical ambiguities found.")

    st.subheader("2️⃣ Structural Disambiguation")
    st.info(structural_disambiguation(text))

    st.subheader("3️⃣ Semantic Frame Representation")
    frame = semantic_frame(text)
    st.json(frame)

    st.subheader("4️⃣ UNL-like Representation")
    unl = to_unl(frame)
    if unl:
        st.code("\n".join(unl))
    else:
        st.write("Unable to generate UNL representation.")

    st.subheader("5️⃣ Semantic Graph")
    if unl:
        G = nx.DiGraph()
        for relation in unl:
            rel, args = relation.split("(")
            arg1, arg2 = args.strip(")").split(",")
            G.add_edge(arg1, arg2, label=rel)
        plt.figure(figsize=(5, 3))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightblue", font_size=10)
        labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        st.pyplot(plt)

import streamlit as st
import google.generativeai as genai
import fitz  
from docx import Document
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def get_gemini_title(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Based on the following content, generate a **catchy, engaging, and well-structured** title that "
        f"grabs attention and clearly represents the main idea:\n\n{text[:2000]}\n\n"
        "Ensure the title is **one** and it has to be concise, compelling, and naturally encourages readers to explore the topic."
    )
    response = model.generate_content(prompt)
    return response.text.strip()
def get_gemini_keywords(title):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Generate 10 to 15 relevant keywords based on the following title:\n\n"
        f"Title: {title}\n\n"
        "Provide the keywords as a *comma-separated* list without any numbering."
    )
    response = model.generate_content(prompt)
    return list(set(response.text.strip().split(", ")))

# Generate three well-structured articles
def generate_articles():
    keywords_str = ", ".join(st.session_state.keywords)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""Write **three engaging and well-structured articles** on **{st.session_state.title}**.  
    **1. Hook the Reader**  
    Start with a **bold claim, surprising fact, or compelling question** to grab attention instantly.  
    **2. Make It Relatable**  
    Use **analogies, storytelling, or real-world examples** to simplify complex ideas.  
    **3. Ensure Readability**  
    - Use **clear subheadings** to organize content.  
    - Keep paragraphs **short and to the point** for easy skimming.  
    - Maintain **smooth transitions** between sections.  
    **4. Simplify Complexity**  
    - Break down **abstract ideas** using analogies.  
    - Use **research-backed insights** to add credibility.  
    **5. Balance Depth & Clarity**  
    - Keep the tone **conversational yet professional** (avoid jargon overload).  
    - Offer **actionable takeaways** that provide value.  
    - Naturally integrate **{keywords_str}** into the content.  
    **6. Drive Action**  
    - Highlight the **real-world impact** of the topic.  
    - Discuss **future implications** and industry trends.  
    - End with a **strong call to action** to spark engagement.  
     **7. Keep It Concise**  
    Each article should be **under 500 words** for maximum engagement.  
    **Formatting Tip:** Separate each article with:  
    `---ARTICLE END---`
    """
    response = model.generate_content(prompt)
    articles = response.text.strip().split("---ARTICLE END---")
    st.session_state.articles = [article.strip() for article in articles if article.strip()]
    st.rerun()

# Generate content based on the selected article 
def generate_content(content_type, selected_article):
    keywords_str = ", ".join(st.session_state.keywords)
    model = genai.GenerativeModel("gemini-1.5-flash")
    if content_type == "Blog":
        prompt = f"""
        Write a **300-400 word blog post** for your target audience on **{st.session_state.title}**.
        ### Key Guidelines:
        - **Hook the Reader** – Start with a bold statement or surprising fact to grab attention.
        - **Engaging & Clear Structure** – Use **subheadings, bullet points, and short paragraphs** for readability.
        - **Fresh Insights** – Focus on unique perspectives, emerging trends, and real-world impact.
        - **Conversational Tone** – Keep it **friendly, direct, and jargon-free**, addressing the reader as “you.”
        - **Credibility & Examples** – Back insights with **data, expert opinions, or real-life examples** to add authority.
        - **SEO Optimization** – Naturally integrate relevant **keywords** and **internal links** for better visibility.
        - **Call to Action** – End with a strong **question or discussion prompt** to encourage engagement.
        ### Enhancements for Impact:
        1. **Stronger Headings** - Keep them concise, clear, and engaging.
        2. **Smooth Transitions** – Ensure a natural flow between sections.
        3. **Compelling CTA** – Encourage readers to **share opinions and join the discussion.**
        4. **Relatable Examples** – Use real-life scenarios to make complex topics accessible.
        Make it insightful, engaging, and easy to read!
        """
    elif content_type == "LinkedIn":
       prompt = (
        f"Write a compelling **300-word** LinkedIn post on **{st.session_state.title}** that grabs attention and makes people *stop scrolling*. 💥 Ensure it's **engaging**, **thought-provoking**, and **encourages interaction**."
        f" **Hook:** Start with a bold statement, surprising fact, or a relatable question (1 line).\n"
        f" **Make It Skimmable:** Use short sentences, line breaks, and bold key points for easy reading.\n"
        f" **Explain Simply:** Describe the concept in a crisp, easy-to-understand way (1 line).\n"
        f" **Add a Quick Analogy or Example:** Make it relatable with a simple comparison (1 line).\n"
        f" **Call to Action:** End with a thought-provoking question to spark discussion (1 line).\n"
        f" **Use Hashtags:** Add relevant hashtags at the end.\n\n"
        f" **Tone:**\n"
        f"content has to be like human written and more crisp short etc"
        f"- Engaging, simple, and beginner-friendly.\n"
        f"- Short, punchy, and easy to skim.\n"
        f"- Informative with a touch of wit—just enough to make it interesting!"
    )
    elif content_type == "Twitter":
        prompt = f"""
        Write a short, engaging tweet on **{st.session_state.title}** (max 280 chars).  
        - Keep it **under 240 chars**, natural tone (no all caps).  
        - **Front-load key message** in the first 3-4 words.  
        - Spark **emotion, passion, or excitement**.  
        - Add a **clear CTA** (reply, click, share).  
        - Use **2-5 hashtags & 1 emoji** for reach.  
        - Tag someone or add a **link** if relevant.  
        - Ensure it's **engaging & optimized for interaction**.
        """
    elif content_type == "Email":
        prompt =f"""
        Generate a **catchy subject line** for **{st.session_state.title}**.
        Write an engaging, **scannable email** with:
        - **Compelling hook** (question or bold statement)
        - **Short paragraphs & bullet points** for readability
        - **Inverted pyramid structure** leading to a strong **CTA**
        Use **keywords naturally**: {keywords_str}, avoiding repetition.  
        Ensure a **smooth, conversational flow**.  
        Write a **clear, action-driven CTA** that encourages interaction.  
        Optimize for **mobile readability** & **avoid spam triggers**.  
        Keep it **engaging, relevant & thought-provoking**.
        """ 
    response = model.generate_content(prompt)
    st.session_state.generated_content = response.text.strip()
    st.rerun()
st.title("AI-Powered Article Generator")
uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
if "title" not in st.session_state:
    st.session_state.title = None
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "articles" not in st.session_state:
    st.session_state.articles = []
if "generated_content" not in st.session_state:
    st.session_state.generated_content = ""
if "prev_uploaded_file" not in st.session_state:
    st.session_state.prev_uploaded_file = None

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.prev_uploaded_file:
        st.session_state.prev_uploaded_file = uploaded_file.name  

        with st.spinner("Extracting text..."):
            if uploaded_file.name.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                extracted_text = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a .pdf or .docx file.")
                extracted_text = ""

        if extracted_text:
            with st.spinner("Generating title and keywords..."):
                title = get_gemini_title(extracted_text)
                keywords = get_gemini_keywords(title)

            st.session_state.title = title
            st.session_state.keywords = keywords

    if st.session_state.title:
        st.subheader("Suggested Title:")
        st.write(f"**{st.session_state.title}**")

    st.subheader("Suggested Keywords:")
    
    cols = st.columns(4)
    keywords_to_remove = []

    for i, kw in enumerate(st.session_state.keywords):
        with cols[i % 4]: 
            if st.button(f"ˣ {kw}", key=f"del_{kw}"):
                keywords_to_remove.append(kw) 
                
    if keywords_to_remove:
        st.session_state.keywords = [kw for kw in st.session_state.keywords if kw not in keywords_to_remove]
        st.rerun()

    custom_key = f"custom_keyword_{len(st.session_state.keywords)}"
    custom_keyword = st.text_input("Add custom keyword", key=custom_key, placeholder="Enter a keyword")

    if st.button("Add Keyword"):
        if custom_keyword.strip() and custom_keyword.strip() not in st.session_state.keywords:
            st.session_state.keywords.append(custom_keyword.strip())  
            st.rerun()

    st.subheader("Generate Articles")
    if st.button("Generate Articles"):
        generate_articles()
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None
if "selected_article" not in st.session_state:
    st.session_state.selected_article = None

if st.session_state.articles:
    st.subheader("Generated Articles:")

    for i, article in enumerate(st.session_state.articles):
        title = article.splitlines()[0] 
        is_selected = st.session_state.selected_index == i
        article_style = "background-color: #F1C1C0; padding: 7px; border-radius: 5px;" if is_selected else ""
        cols = st.columns([0.05, 0.95])
        with cols[0]:
            if st.button("🔴" if is_selected else "⚪", key=f"select_{i}"):
                st.session_state.selected_index = i 
                st.session_state.selected_article = article
                st.rerun()  
        with cols[1]:
            st.markdown(f"<div style='{article_style}'><b>{title}</b></div>", unsafe_allow_html=True)
        st.write(article)
        st.divider()  
    if st.session_state.selected_article:
        st.subheader("Generate Content")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Blog Content"):
                generate_content("Blog", st.session_state.selected_article)
        with col2:
            if st.button("LinkedIn Content"):
                generate_content("LinkedIn", st.session_state.selected_article)
        with col3:
            if st.button("Twitter Content"):
                generate_content("Twitter", st.session_state.selected_article)
        with col4:
            if st.button("Email Content"):
                generate_content("Email", st.session_state.selected_article)
        if st.session_state.get("generated_content"):
            st.subheader("Generated Content:")
            st.write(st.session_state.generated_content)

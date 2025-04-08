import sys
import sqlite3
import os
import time
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Patch for sqlite version if needed
if sqlite3.sqlite_version_info < (3, 35, 0):
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        from chromadb.utils import embedding_functions
        embedding_functions._sqlite3 = sqlite3
        sys.modules['sqlite3'] = sqlite3

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Please create a .env file with your GOOGLE_API_KEY")
    st.stop()

# Set default session state
default_states = {
    "keywords": [], "title": "", "show_keyword_section": False, "all_chunks": [],
    "selected_audience": "General Public", "analysis_result": "", "generated_article": "",
    "refinement_text": "", "refined_article": "", "current_article": "",
    "generated_post": "", "post_type": None, "selected_tone": None,
    "custom_tone": "", "humanized_content": ""
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.set_page_config(page_title="AI Post Generator", layout="centered")
st.title("Post Generator")

@st.cache_data(show_spinner=False)
def pdf_to_limited_chunks(pdf_file, chunk_size=700, chunk_overlap=100):
    """Extract text from PDF and split into chunks (first 5 returned)"""
    try:
        reader = PdfReader(pdf_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = splitter.split_text(text)
        st.session_state.all_chunks = all_chunks
        return all_chunks[:10]  # Limit initial processing
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def get_llm(temperature=0.5, model="gemini-1.5-flash"):
    return ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=temperature)

def generate_title(chunks):
    """Generate a more engaging, insightful title from document content."""
    if not chunks:
        return ""

    llm = get_llm(temperature=0.9) 
    selected_chunks = chunks[:5] + chunks[-5:] if len(chunks) >= 10 else chunks
    combined_text = "\n\n".join(selected_chunks)

    prompt = f"""
    You are a professional content strategist.

    Based on the following document excerpts, generate ONE **engaging and clear** article title.

    Content:
    {combined_text}

    🔹 The title should:
    - Be 6–12 words max
    - Capture the **main problem or opportunity**
    - Use strong, simple language (no fluff or jargon)
    - Feel like a **headline** you'd want to click
    Examples of good titles:
    - "Why Smart Cities May Fail Without AI (2024)"
    - "How Climate Startups Are Saving the Planet"
    - "The Truth About Data Privacy in 2025"

    Return ONLY the final title text.
    """

    result = llm.invoke(prompt).content.strip()
    return result.split('\n')[0].strip('"').strip()


def generate_keywords(title, audience):
    """Generate keywords for SEO based on audience"""
    llm = get_llm(temperature=0.5)
    prompt = f"""
    Generate 15 relevant keywords for this title targeting {audience}:
    Title: {title}

    - Use short words/phrases
    - Audience-specific
    - SEO optimized
    - Comma-separated list only
    """
    response = llm.invoke(prompt).content
    return [kw.strip() for kw in response.split(",") if kw.strip()][:15]


def analyze_keywords(keywords, audience):
    """Return ranked, bulleted, single-line keyword analysis with numeric relevance ratings"""
    llm = get_llm(temperature=0.3)
    prompt = f"""
    Analyze the following keywords for the audience: {audience}.
    For each keyword:
    Write a short explanation of how it helps or relates to the audience.
    Rate its relevance as a percentage (e.g., 100% = most relevant).
    If the keyword is irrelevant, give it 0%.
    Format each result as a bullet point on a single line like this:
    keyword: explanation. – Rating (e.g., 100%)
    Then, sort all keywords by rating in descending order (highest rated first). Do not group them—just a flat, sorted bullet list.
    Keywords: {", ".join(keywords)}
    """
    return llm.invoke(prompt).content

def generate_article(title, keywords):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="C:/Users/MATHALIN/smartsuggestion/chroma_db"
        )
        query = f"{title}. Keywords: {', '.join(keywords)}"
        relevant_docs = vector_store.similarity_search(query, k=5)
        relevant_content = "\n\n".join([doc.page_content for doc in relevant_docs])
        llm = get_llm(temperature=0.5)
        prompt = f"""
            Write a clear, engaging article (500–600 words) on: {title}
            Make it simple, crisp, and easy to follow for a broad audience.
            **STRUCTURE & STYLE**
            1. **Introduction** – Start with a bold claim, surprising stat, or thought-provoking question.
            2. **Body** – Use short paragraphs and clear subheadings. Include:
            - Big Picture (why it matters)
            - Practical Impacts (real-world relevance)
            - Simplified Technical Insights
            3. **Content Quality**
            - Use analogies and real examples
            - Include 2–3 relevant facts or stats
            - Naturally weave in keywords: {', '.join(keywords)}
            4. **Tone**
            - Professional yet conversational
            - No jargon, no fluff
            5. **Conclusion**
            - Summarize key points
            - Share future implications or prompt reflection

            Reference this content:
            {relevant_content}
        """
        return llm.invoke(prompt).content.strip()

    except Exception as e:
        st.error(f"Failed to generate article: {str(e)}")
        return "Error: Unable to generate article."

def generate_social_post(article_content, post_type, tone, custom_tone, keywords, audience):
    """Generate social media post based on article content and post type"""
    try:
        selected_tone = tone.split(" ")[0].lower() if tone != "Custom ✏️" else custom_tone.lower()
        temperature = 0.7 if selected_tone == "modern" else 0.5
        llm = get_llm(temperature=temperature)
        max_content_length = 2000
        article_preview = article_content[:max_content_length] + ("..." if len(article_content) > max_content_length else "")
        limited_keywords = keywords[:5] if len(keywords) > 5 else keywords
        
        post_prompts = {
            "blog": f"""
                Write a **300-400 word blog post** based on this content for {audience}:
                {article_preview}
                
                ### Key Guidelines:
                - **Tone:** {selected_tone.upper()} ({custom_tone if tone == "Custom ✏️" else tone})
                - **Hook the Reader** – Start with a bold statement or surprising fact
                - **Engaging Structure** – Use subheadings, bullet points, and short paragraphs(5 lines)
                - **Fresh Insights** – Focus on unique perspectives and real-world impact
                - **Conversational Style** – Keep it {selected_tone} and jargon-free
                - **Credibility** – Back insights with data or examples
                - **SEO Optimization** – Use keywords: {', '.join(limited_keywords)}
                - **Call to Action** – End with a discussion prompt
                -**content has to be like human written and more crisp short etc*
                ### Tone-Specific Enhancements:
                {"- Use emojis and casual language" if selected_tone == "casual" else ""}
                {"- Maintain professional terminology" if selected_tone == "formal" else ""}
              {"- Keep the tone fresh, clear, and up-to-date" if selected_tone == "modern" else ""}

                {"- Follow custom tone description exactly" if tone == "Custom " else ""}
                
                Return ONLY the formatted blog post.
                """,
                
            "linkedin": f"""
                Write a compelling **300-word LinkedIn post** based on this content that grabs attention and makes people *stop scrolling*. 💥 Ensure it's **engaging**, **thought-provoking**, and **encourages interaction**.
                {article_preview}
                
                **Requirements:**
                - Tone: {selected_tone.upper()} ({custom_tone if tone == "Custom ✏️" else tone})
           
                -**Hook:** Start with a bold statement, surprising fact, or a relatable question (1 line).\n
                -**Make It Skimmable:** Use short sentences, line breaks, and bold key points for easy reading.\n
                -**Explain Simply:** Describe the concept in a crisp, easy-to-understand way (1 line).\n
                -**Add a Quick Analogy or Example:** Make it relatable with a simple comparison (1 line).\n
                -**Call to Action:** End with a thought-provoking question to spark discussion (1 line).\n
                -**Use Hashtags:** Add relevant hashtags at the end.\n\n
                -**Tone:**\n
                -**content has to be like human written and more crisp short etc**
                - Engaging, simple, and beginner-friendly.\n
                - Short, punchy, and easy to skim.\n
                - Informative with a touch of wit—just enough to make it interesting!
                        
                **Tone Guidelines:**
                {"- Casual, friendly, with emojis" if selected_tone == "casual" else ""}
                {"- Professional but engaging" if selected_tone == "formal" else ""}
               {"- Keep the tone fresh, clear, and up-to-date" if selected_tone == "modern" else ""}

                {"- Custom: " + custom_tone if tone == "Custom " else ""}
                
                Return ONLY the LinkedIn post content.
                """,
                
            "twitter": f"""
                Write an **one** engaging tweet thread based on this content:
                {article_preview}
                
                **Requirements:**
                - Tone: {selected_tone.upper()} ({custom_tone if tone == "Custom ✏️" else tone})
                - Keep it **under 240 chars**, natural tone (no all caps).  
                - **Front-load key message** in the first 3-4 words.  
                - Spark **emotion, passion, or excitement**.  
                - Add a **clear CTA** (reply, click, share).  
                - Use **2-5 hashtags & 1 emoji** for reach.  
                - Tag someone or add a **link** if relevant.  
                - Ensure it's **engaging & optimized for interaction**.
                - **content has to be like human written and more crisp short etc*
                **Tone Guidelines:**
                {"- Casual, conversational" if selected_tone == "casual" else ""}
                {"- Professional but concise" if selected_tone == "formal" else ""}
                {"- Keep the tone fresh, clear, and up-to-date" if selected_tone == "modern" else ""}

                {"- Custom: " + custom_tone if tone == "Custom " else ""}
                """,
                
            "email": f"""
                Write a professional email based on this content:
                {article_preview}
                
                **Requirements:**
                - Tone: {selected_tone.upper()} ({custom_tone if tone == "Custom ✏️" else tone})
                -**Make It More Personalized eg("  Dear [Name],")
                Write an engaging, **scannable email** with:
                - **Compelling hook** (question or bold statement)
                - **Short paragraphs & bullet points** for readability
                - **Inverted pyramid structure** leading to a strong **CTA**
                Use **keywords naturally**:  avoiding repetition.  
                Ensure a **smooth, conversational flow**.  
                Write a **clear, action-driven CTA** that encourages interaction.  
                Optimize for **mobile readability** & **avoid spam triggers**.  
                Keep it **engaging, relevant & thought-provoking**.
                **content has to be like human written and more crisp short etc*
                **Tone Guidelines:**
                {"- Friendly and approachable" if selected_tone == "casual" else ""}
                {"- Formal and professional" if selected_tone == "formal" else ""}
               {"- Keep the tone fresh, clear, and up-to-date" if selected_tone == "modern" else ""}

                {"- Custom: " + custom_tone if tone == "Custom " else ""}
                
                Format:
                Subject: [subject line]
                
                [email body]
                
                Return ONLY the email content.
                """
        }
        
        prompt = post_prompts.get(post_type, "")
        return llm.invoke(prompt).content if prompt else None
        
    except Exception as e:
        st.error(f"Error generating post: {str(e)}")
        return None

def refine_article(current_article, refinement_instruction, keywords):
    """Refine article based on user instructions"""
    try:
        llm = get_llm(temperature=0.4)
        limited_keywords = keywords[:5] if len(keywords) > 5 else keywords
        
        refine_prompt = f"""
        Please refine the following article based on these specific instructions:
        
        REFINEMENT REQUEST:
        {refinement_instruction}
        
        CURRENT ARTICLE CONTENT:
        {current_article}
        
        GUIDELINES FOR REFINEMENT:
        1. Make only the requested changes - don't modify other parts
        2. Keep the same overall structure and tone
        3. Maintain all key facts and information
        4. Preserve the keyword integration: {', '.join(limited_keywords)}
        5. Highlight changes by bolding new or modified text
        
        OUTPUT REQUIREMENTS:
        - Return the complete revised article
        - Mark changes in bold
        - Keep the same approximate length
        - Maintain all original section headings
        """
        
        return llm.invoke(refine_prompt).content
    except Exception as e:
        st.error(f"Error refining article: {str(e)}")
        return None
    
def humanize_content(content):
    """Make content sound more like it was written by a human"""
    try:
        llm = get_llm(temperature=0.7)
        
        humanize_prompt = f"""
            Rewrite the content below so it sounds natural, human, and engaging—like something you'd say to a friend. 
            Keep the original message, but make it flow effortlessly with personality and a conversational tone.
            Guidelines:
            - Mix short, punchy lines with longer ones.
            - Use smooth, natural transitions.
            - Add light personality or perspective where it fits.
            - Avoid sounding robotic, repetitive, or overly formal.
            - No fluff. No filler. Just clean, real-sounding writing.
            Don’t:
            - Mention it's rewritten.
            - Add or change facts.
            Return only the rewritten version—no extra notes.
            """
        if post_types == "twitter":
            prompt = f"""
            { humanize_prompt}
            ### Format: Twitter Post
                - Keep it **under 280 characters**
                - Use casual, punchy phrasing
                - Add appropriate **emojis** if it fits
                - Use up to **2–3 relevant hashtags**
                - Make it scroll-stopping and bold
                - Avoid long intros or conclusions
                - Focus on **one big takeaway or hook**
            Content:
            {content}
            Return ONLY the final tweet text. No intro, no notes.
            """
        elif post_types == "linkedin":
            prompt = f"""
            { humanize_prompt}
            ### Format: LinkedIn Post
            - Professional yet friendly
            - Add a personal perspective or insight
            - Break into short paragraphs (2–3 lines max)
            - Add 2–5 relevant hashtags at the end
            - Length: ~150–300 words max
            - No fluff—be informative and relatable
            Content:
            {content}
            Return ONLY the rewritten LinkedIn post.
            """
        elif post_types == "email":
            prompt = f"""
            { humanize_prompt}
            ### Format: Email Body
            - Conversational and helpful tone
            - Clear intro, body, and closing
            - Add transitions like you're writing to a colleague or friend
            - End with a CTA or thoughtful note
            - Length: 150–300 words
            - No hashtags or emojis
            Content:
            {content}
            Return ONLY the rewritten email body.
            """
        else:
            prompt = f"""
            { humanize_prompt}
            Content:
            {content}
            Return ONLY the rewritten version with a natural, human tone.
            """
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error during humanization: {str(e)}"
    except Exception as e:
        st.error(f"Error humanizing content: {str(e)}")
        return None
    
def reset_state_after(state_to_keep):
    """Reset state variables after certain operations"""
    states_to_reset = {
        "title": ["analysis_result", "generated_article", "refined_article", 
                 "current_article", "generated_post"],
        "keywords": ["analysis_result", "generated_article", "refined_article", 
                    "current_article", "generated_post"],
    }
    for state_to_reset in states_to_reset.get(state_to_keep, []):
        st.session_state[state_to_reset] = ""
    if state_to_keep == "keywords":
        st.session_state.post_type = None

# Main App UI
uploaded_file = st.file_uploader("Upload the document", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        selected_chunks = pdf_to_limited_chunks(uploaded_file)

    if selected_chunks:
        if st.button("Generate Title"):
            try:
                title = generate_title(selected_chunks)
                st.session_state.title = title
                st.session_state.show_keyword_section = True
            except Exception as e:
                st.error(f"Error generating title: {str(e)}")

    if st.session_state.title:
        st.markdown(f"## {st.session_state.title}")

        if st.session_state.show_keyword_section:
            st.divider()

            st.session_state.selected_audience = st.selectbox(
                "Select Target Audience",
                ["General Public", "Business Leaders", "Policy Makers", 
                 "Investors", "Media", "Sales Teams", "Marketing Professionals"],
                index=0
            )

            if st.button("Generate Keywords"):
                with st.spinner("Generating keywords..."):
                    st.session_state.keywords = generate_keywords(
                        st.session_state.title, 
                        st.session_state.selected_audience
                    )
                    st.session_state.analysis_result = analyze_keywords(
                    st.session_state.keywords,
                    st.session_state.selected_audience
                )
            if st.session_state.get("update_analysis", False):
                st.session_state.analysis_result = analyze_keywords(
                    st.session_state.keywords,
                    st.session_state.selected_audience
                )
                st.session_state.update_analysis = False
            if st.session_state.keywords:
                st.subheader("Keywords")
                cols = st.columns(4)
                keywords_to_remove = []
                for i, kw in enumerate(st.session_state.keywords):
                    with cols[i % 4]:
                        if st.button(f"ˣ {kw}", key=f"del_{kw}"):
                            keywords_to_remove.append(kw)

                if keywords_to_remove:
                    st.session_state.keywords = [
                        kw for kw in st.session_state.keywords if kw not in keywords_to_remove
                    ]
                    st.session_state.update_analysis = True
                    reset_state_after("keywords")
                    st.rerun()
                with st.form("add_keyword_form"):
                    custom_keyword = st.text_input(
                        "Add custom keyword",
                        key=f"custom_keyword_{len(st.session_state.keywords)}",
                        placeholder="Enter a keyword and press Enter"
                    )
                    submitted = st.form_submit_button("Add Keyword")

                    if submitted or custom_keyword:
                        new_kw = custom_keyword.strip()
                        if new_kw and new_kw not in st.session_state.keywords:
                            st.session_state.keywords.append(new_kw)
                            st.session_state.update_analysis = True
                            reset_state_after("keywords")
                            st.rerun()
                        elif new_kw:
                            st.warning("Keyword already exists in the list")

                st.divider()
                if st.session_state.analysis_result:
                    st.subheader("Keyword Relevance Analysis")
                    st.markdown(st.session_state.analysis_result)
                st.divider()
                if st.button("Generate Article"):
                    with st.spinner("Generating article..."):
                        article = generate_article(
                            st.session_state.title,
                            st.session_state.keywords
                        )
                        if article:
                            st.session_state.generated_article = article
                            st.session_state.current_article = article
                            st.session_state.refined_article = ""  
                            st.success(" Article generated!")

                if st.session_state.current_article:
                    article_type = "Refined Article" if st.session_state.refined_article and st.session_state.current_article == st.session_state.refined_article else "Generated Article"
                    st.subheader(article_type)
                    st.markdown(st.session_state.current_article)

                    st.divider()
                    st.subheader(" Refine Article")
                    with st.form("refinement_form"):
                        user_input = st.text_input(
                            "Enter refinement instructions",
                            value=st.session_state.refinement_text,
                            placeholder="What would you like to change in the article?",
                            key=f"refinement_input_{hash(st.session_state.current_article)}"
                        )
                        submitted = st.form_submit_button("Refine Article")

                    if submitted:
                        if user_input.strip():
                            with st.spinner("Refining article..."):
                                refined = refine_article(
                                    st.session_state.current_article,
                                    user_input,
                                    st.session_state.keywords
                                )
                                if refined:
                                    st.session_state.refined_article = refined
                                    st.session_state.current_article = refined
                                    st.session_state.refinement_text = ""
                                    st.success(" Article refined!")
                                    st.rerun()
                        else:
                            st.warning("Please enter refinement instructions.")
                    else:
                        st.session_state.refinement_text = user_input

                    if st.session_state.generated_article and st.session_state.refined_article:
                        st.divider()
                        st.subheader("Choose Version")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Use Original Article"):
                                st.session_state.current_article = st.session_state.generated_article
                                st.session_state.refinement_text = ""
                                st.rerun()
                        with col2:
                            if st.button("Use Refined Article"):
                                st.session_state.current_article = st.session_state.refined_article
                                st.session_state.refinement_text = ""
                                st.rerun()


                #post type
                    st.divider()
                    st.subheader("Convert to Social Post")
                    st.subheader("Select Post Type")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    post_types = {
                        "blog": "Blog Post",
                        "linkedin": "LinkedIn Post",
                        "twitter": "Twitter",
                        "email": "Email Post"
                    }
                    
                    cols = [col1, col2, col3, col4]
                    for i, (key, label) in enumerate(post_types.items()):
                        with cols[i]:
                            if st.button(f" {label}", key=f"{key}_post_btn", use_container_width=True):
                                st.session_state.post_type = key
                                st.rerun()
                    if st.session_state.post_type:
                        st.subheader("Select Tone")
                        tone = st.radio(
                            "Choose writing style:",
                            ["Casual ", "Formal ", "modern ", "Custom "],
                            horizontal=True,
                            key="tone_selection",
                            label_visibility="collapsed"
                        )
                        
                        st.session_state.selected_tone = tone

                        if tone == "Custom ":
                            st.session_state.custom_tone = st.text_input(
                                "Describe the tone you want (e.g., 'Motivational', 'Technical', 'Friendly expert')", 
                                key="custom_tone_input"
                            )
                        
                        if st.button(f"Generate {st.session_state.post_type.replace('_', ' ').title()} Post"):
                            with st.spinner(f"Crafting your {st.session_state.post_type} post..."):
                                post_content = generate_social_post(
                                    st.session_state.current_article,
                                    st.session_state.post_type,
                                    st.session_state.selected_tone,
                                    st.session_state.custom_tone if tone == "Custom " else "",
                                    st.session_state.keywords,
                                    st.session_state.selected_audience
                                )
                                
                                if post_content:
                                    st.session_state.generated_post = post_content
                                    st.rerun()
                        if st.session_state.generated_post:
                            st.subheader(f"Your {st.session_state.post_type.title()} Post")
                            st.markdown(st.session_state.generated_post)
                        if st.button(" Humanize Post", help="Make the post sound more naturally human-written"):
                                with st.spinner("Making post sound more human..."):
                                    humanized_post = humanize_content(st.session_state.generated_post)
                                    if humanized_post:
                                        st.session_state.generated_post = humanized_post
                                        st.success("Post has been humanized!")
                                        st.rerun()
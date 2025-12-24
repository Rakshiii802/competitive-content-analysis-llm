import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

st.set_page_config(page_title="Content Gap Analyzer")

st.title("Competitive Content Analysis (LLM-based)")

st.markdown(
    """
This tool compares brand content with competitor content
and highlights missing topics and improvement areas.
"""
)

    brand_text = st.text_area(
    "Brand Page Content (Sample)",
    height=220,
    value="""
We help online brands grow using data-driven digital marketing.

Our services include SEO optimization, landing page improvements,
conversion tracking, and marketing automation to increase sales
and customer retention.
"""
)



competitor_text = st.text_area(
    "Paste Competitor Page Content",
    height=200,
    value="""
We work with modern e-commerce companies to scale growth.

Our approach focuses on content strategy, SEO-led demand generation,
paid media optimization, and lifecycle marketing for long-term growth.
"""

)

def analyze_content(brand, competitor):
    texts = [brand, competitor]

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents("SEO content comparison")

    context = ""
    for i, d in enumerate(docs):
        context += f"\nContent {i+1}:\n{d.page_content[:1500]}\n"

    llm = OpenAI(temperature=0.3)

    prompt = f"""
You are evaluating website content from an SEO perspective.

Brand content:
{brand[:1500]}

Competitor content:
{competitor[:1500]}

Using the above information, answer:
1. What important topics are missing in the brand content?
2. What areas can be improved for better SEO?
3. Any structural or clarity issues?

Be concise and practical.
"""

    return llm(prompt)

if st.button("Run Analysis"):
    if not brand_text or not competitor_text:
        st.warning("Please provide both brand and competitor content.")
    else:
        with st.spinner("Analyzing content..."):
            result = analyze_content(brand_text, competitor_text)
        st.subheader("Analysis Result")
        st.write(result)


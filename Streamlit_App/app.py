import streamlit as st

from app_display_results import display_resume_analysis
from app_sidebar import sidebar
from llm_functions import instantiate_LLM_main, get_api_keys_from_local_env
from resume_analyzer import resume_analyzer_main
from retrieval import retrieval_main


def main():
    """Analyze the uploaded resume."""
    uploaded_file = st.session_state.get("uploaded_file", None)

    if not uploaded_file:
        st.warning("Please upload a resume file before analysis.")
        return

    if st.button("Analyze resume"):
        with st.spinner("Please wait..."):
            try:
                # 1. Create the Langchain retrieval
                retrieval_main()

                # 2. Instantiate a deterministic LLM with a temperature of 0.0.
                st.session_state.llm = instantiate_LLM_main(temperature=0.0, top_p=0.95)

                # 3. Instantiate LLM with temperature >0.1 for creativity.
                st.session_state.llm_creative = instantiate_LLM_main(
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                )

                # 4. Analyze the resume
                if "documents" not in st.session_state or not st.session_state.documents:
                    st.error("No documents available for analysis.")
                    return

                st.session_state.SCANNED_RESUME = resume_analyzer_main(
                    llm=st.session_state.llm,
                    llm_creative=st.session_state.llm_creative,
                    documents=st.session_state.documents,
                )

                # 5. Display results
                display_resume_analysis(st.session_state.SCANNED_RESUME)

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # 1. Set app configuration
    st.set_page_config(page_title="Resume Scanner", page_icon="🚀")
    st.title("🔎 Resume Scanner")

    # 2. Initialize session states
    st.session_state.setdefault("temperature", 0.7)
    st.session_state.setdefault("top_p", 0.95)
    st.session_state.setdefault("documents", None)

    # 3. Get API keys from local "keys.env" file
    openai_api_key, zhipu_api_key, qwen_api_key, deepseek_api_key = get_api_keys_from_local_env()

    # 4. Create the sidebar
    sidebar(openai_api_key, zhipu_api_key, qwen_api_key, deepseek_api_key)

    # 5. File uploader widget
    st.session_state.uploaded_file = st.file_uploader(
        label="**Upload Resume**",
        accept_multiple_files=False,
        type=(["pdf"]),
    )

    # 6. Analyze the uploaded resume
    main()

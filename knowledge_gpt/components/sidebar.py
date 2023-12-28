import streamlit as st

from knowledge_gpt.components.faq import faq
from dotenv import load_dotenv
import os

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
            "2. Upload a pdf, docx, or txt file📄\n"
            "3. Ask a question about the document💬\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )

        temperature = st.number_input(
                            "Randomness",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,  # Default value
                            help="Set the temperature value (between 0 and 1)."
                        )

        st.session_state["OPENAI_API_KEY"] = api_key_input
        st.session_state["temperature"] = temperature

        # st.markdown("---")
        # st.markdown("# About")
        # st.markdown(
        #     "📖KnowledgeGPT allows you to ask questions about your "
        #     "documents and get accurate answers with instant citations. "
        # )
        # st.markdown(
        #     "This tool is a work in progress. "
        #     "You can contribute to the project on [GitHub](https://github.com/mmz-001/knowledge_gpt) "  # noqa: E501
        #     "with your feedback and suggestions💡"
        # )
        # st.markdown("Made by [mmz_001](https://twitter.com/mm_sasmitha)")
        # st.markdown("---")

        # faq()

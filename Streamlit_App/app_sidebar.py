import streamlit as st

from app_constants import list_Assistant_Languages, list_LLM_providers, LLM_CONFIGS


def expander_model_parameters(LLM_provider, api_key, config):
    """动态添加 API Key 和模型参数的 Streamlit 界面"""
    # 初始化默认值
    st.session_state.setdefault("LLM_provider", LLM_provider)
    st.session_state.setdefault("temperature", 0.7)
    st.session_state.setdefault("top_p", 0.95)
    st.session_state.setdefault("selected_model", config["models"][0])

    # API Key 输入框
    st.session_state.api_key = st.text_input(
        f"{config['api_key_label']} - [Get an API key]({config['api_key_link']})",
        value=api_key,
        type="password",
        placeholder="Insert your API key",
    )

    # 模型选择和参数配置
    with st.expander("**Models and parameters**", expanded=False):
        st.session_state.selected_model = st.selectbox(
            f"Choose {LLM_provider} model", config["models"]
        )
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
        )
        st.session_state.top_p = st.slider(
            "Top-p",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.05,
        )


def sidebar(openai_api_key, zhipu_api_key, qwen_api_key, deepseek_api_key):
    """创建侧边栏"""
    with st.sidebar:
        st.caption("🚀 A resume analysis tool powered by 🔗 Langchain")
        st.write("")

        # 提供商选择绑定到 session_state.LLM_provider
        st.session_state.setdefault("LLM_provider", list_LLM_providers[0])
        llm_chooser = st.radio(
            "Select provider",
            list_LLM_providers,
            index=list_LLM_providers.index(st.session_state.LLM_provider),
            key="LLM_provider",
        )

        st.divider()

        # 动态加载提供商的参数界面
        if llm_chooser in LLM_CONFIGS:
            config = LLM_CONFIGS[llm_chooser]
            api_key = {
                "OpenAI": openai_api_key,
                "ZhiPu": zhipu_api_key,
                "Qwen": qwen_api_key,
                "DeepSeek": deepseek_api_key,
                "Ollama": "deepseek_api_key",
            }.get(llm_chooser, "")
            expander_model_parameters(llm_chooser, api_key, config)

        # 助手语言选择
        st.divider()
        st.session_state.assistant_language = st.selectbox(
            "Assistant language", list_Assistant_Languages
        )

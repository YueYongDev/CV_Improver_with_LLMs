import os

import streamlit as st
# dotenv and os
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
# LLM: openai
from langchain_openai import ChatOpenAI


def get_api_keys_from_local_env():
    """Get OpenAI, Gemini and Cohere API keys from local .env file"""
    try:
        found_dotenv = find_dotenv("keys.env", usecwd=True)
        load_dotenv(found_dotenv)
        try:
            openai_api_key = os.getenv("api_key_openai")
        except:
            openai_api_key = ""
        try:
            zhipu_api_key = os.getenv("api_key_zhipu")
        except:
            zhipu_api_key = ""
        try:
            qwen_api_key = os.getenv("api_key_qwen")
        except:
            qwen_api_key = ""
        try:
            deepseek_api_key = os.getenv("api_key_deepseek")
        except:
            deepseek_api_key = ""
    except Exception as e:
        print(e)

    return openai_api_key, zhipu_api_key, qwen_api_key, deepseek_api_key


def instantiate_LLM(
        LLM_provider, api_key, temperature=0.5, top_p=0.95, model_name=None
):
    """Instantiate LLM in Langchain.
    Parameters:
        LLM_provider (str): the LLM provider; in ["OpenAI","Google"]
        model_name (str): in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-turbo-preview","gemini-pro"].
        api_key (str): google_api_key or openai_api_key
        temperature (float): Range: 0.0 - 1.0; default = 0.5
        top_p (float): : Range: 0.0 - 1.0; default = 1.
    """
    if LLM_provider == "OpenAI":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            model_kwargs={"top_p": top_p},
        )
    if LLM_provider == "ZhiPu":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            temperature=temperature,
            model_kwargs={"top_p": top_p},
        )
    if LLM_provider == "Qwen":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=temperature,
            model_kwargs={"top_p": top_p},
        )
    if LLM_provider == "DeepSeek":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            model_kwargs={"top_p": top_p},
        )
    if LLM_provider == "Ollama":
        llm = ChatOllama(
            api_key=api_key,
            base_url="https://ollama.liangyueyong.cn",
            model=model_name,
            temperature=0.8,
            num_predict=256,
        )

    return llm


def instantiate_LLM_main(temperature, top_p):
    """Instantiate the selected LLM model."""
    print(f"Selected LLM provider: {st.session_state.LLM_provider}")
    print(f"Selected model: {st.session_state.selected_model}")
    print(f"API Key: {st.session_state.api_key}")
    try:
        if st.session_state.LLM_provider == "OpenAI":
            llm = instantiate_LLM(
                "OpenAI",
                api_key=st.session_state.api_key,
                # api_key="sk-9cc8d47328e04d98a7ca430505b6a59c",
                temperature=temperature,
                top_p=top_p,
                model_name=st.session_state.selected_model,
                # model_name="qwen-turbo",
            )
        elif st.session_state.LLM_provider == "ZhiPu":
            llm = instantiate_LLM(
                "ZhiPu",
                api_key=st.session_state.api_key,
                temperature=temperature,
                top_p=top_p,
                model_name=st.session_state.selected_model,
            )
        elif st.session_state.LLM_provider == "Qwen":
            llm = instantiate_LLM(
                "Qwen",
                api_key=st.session_state.api_key,
                temperature=temperature,
                top_p=top_p,
                model_name=st.session_state.selected_model,
            )
        elif st.session_state.LLM_provider == "DeepSeek":
            llm = instantiate_LLM(
                "DeepSeek",
                api_key=st.session_state.api_key,
                temperature=temperature,
                top_p=top_p,
                model_name=st.session_state.selected_model,
            )
        elif st.session_state.LLM_provider == "Ollama":
            llm = instantiate_LLM(
                "Ollama",
                api_key=st.session_state.api_key,
                temperature=temperature,
                top_p=top_p,
                model_name=st.session_state.selected_model,
            )
    except Exception as e:
        st.error(f"An error occured: {e}")
        llm = None
    return llm

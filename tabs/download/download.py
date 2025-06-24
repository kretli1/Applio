import os
import sys
import json
import shutil
import requests
import tempfile
import gradio as gr
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Adiciona diretório atual ao path
now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_download_script
from rvc.lib.utils import format_title
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Limpa diretório temporário do Gradio
gradio_temp_dir = os.path.join(tempfile.gettempdir(), "gradio")
if os.path.exists(gradio_temp_dir):
    shutil.rmtree(gradio_temp_dir)


def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message=i18n("O arquivo enviado não é um modelo válido. Tente novamente.")
        )
    file_name = format_title(os.path.basename(dropbox))
    model_name = file_name
    if ".pth" in model_name:
        model_name = model_name.split(".pth")[0]
    elif ".index" in model_name:
        replacements = ["nprobe_1_", "_v1", "_v2", "added_"]
        for rep in replacements:
            model_name = model_name.replace(rep, "")
        model_name = model_name.split(".index")[0]

    model_path = os.path.join(now_dir, "logs", model_name)
    os.makedirs(model_path, exist_ok=True)
    dest = os.path.join(model_path, file_name)
    if os.path.exists(dest):
        os.remove(dest)
    shutil.move(dropbox, dest)
    gr.Info(i18n(f"{file_name} salvo em {model_path}"))


def fetch_pretrained_data():
    json_url = "https://huggingface.co/IAHispano/Applio/raw/main/pretrains.json"
    pretraineds_custom_path = os.path.join("rvc", "models", "pretraineds", "custom")
    os.makedirs(pretraineds_custom_path, exist_ok=True)
    try:
        with open(os.path.join(pretraineds_custom_path, os.path.basename(json_url)), "r") as f:
            data = json.load(f)
    except Exception:
        try:
            response = requests.get(json_url)
            response.raise_for_status()
            data = response.json()
            with open(os.path.join(pretraineds_custom_path, os.path.basename(json_url)), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            data = {"Titan": {"32k": {"D": "null", "G": "null"}}}
    return data


def get_pretrained_list():
    return list(fetch_pretrained_data().keys())


def get_pretrained_sample_rates(model):
    data = fetch_pretrained_data()
    return list(data.get(model, {}).keys())


def get_file_size(url):
    response = requests.head(url)
    return int(response.headers.get("content-length", 0))


def download_file(url, destination_path, progress_bar):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    response = requests.get(url, stream=True)
    with open(destination_path, "wb") as file:
        for chunk in response.iter_content(1024):
            file.write(chunk)
            progress_bar.update(len(chunk))


def download_pretrained_model(model, sample_rate):
    data = fetch_pretrained_data()
    paths = data[model][sample_rate]
    base_path = os.path.join("rvc", "models", "pretraineds", "custom")
    os.makedirs(base_path, exist_ok=True)
    d_url = f"https://huggingface.co/{paths['D']}"
    g_url = f"https://huggingface.co/{paths['G']}"
    total_size = get_file_size(d_url) + get_file_size(g_url)
    gr.Info(i18n("Baixando modelo pré-treinado..."))
    with tqdm(total=total_size, unit="iB", unit_scale=True) as bar:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as exec:
            exec.submit(download_file, d_url, os.path.join(base_path, os.path.basename(paths['D'])), bar)
            exec.submit(download_file, g_url, os.path.join(base_path, os.path.basename(paths['G'])), bar)
    gr.Info(i18n("Modelo pré-treinado baixado com sucesso!"))


def update_sample_rate_dropdown(model):
    choices = get_pretrained_sample_rates(model)
    return {"choices": choices, "value": choices[0] if choices else None, "__type__": "update"}


def download_tab():
    with gr.Column():
        # Download Model
        gr.Markdown(value=i18n("## Download Model"))
        model_link = gr.Textbox(label=i18n("Model Link"), placeholder=i18n("Cole o link do modelo"), interactive=True)
        model_download_info = gr.Textbox(label=i18n("Output Information"), value="", max_lines=8, interactive=False)
        gr.Button(i18n("Download Model")).click(fn=run_download_script, inputs=[model_link], outputs=[model_download_info])

        # Drop files
        gr.Markdown(value=i18n("## Drop files"))
        dropbox = gr.File(label=i18n("Arraste seu .pth e .index aqui"), type="filepath")
        dropbox.upload(fn=save_drop_model, inputs=[dropbox], outputs=[dropbox])

        # Download Pretrained Models
        gr.Markdown(value=i18n("## Download Pretrained Models"))
        pretrained_model = gr.Dropdown(label=i18n("Pretrained"), choices=get_pretrained_list(), value="Titan", interactive=True)
        pretrained_sample_rate = gr.Dropdown(label=i18n("Sampling Rate"), choices=get_pretrained_sample_rates(pretrained_model.value), value="32k", interactive=True)
        pretrained_model.change(fn=update_sample_rate_dropdown, inputs=[pretrained_model], outputs=[pretrained_sample_rate])
        gr.Button(i18n("Download")).click(fn=download_pretrained_model, inputs=[pretrained_model, pretrained_sample_rate], outputs=[])

        # Baixar Áudio TTS
        gr.Markdown(value=i18n("## Baixar Áudio TTS"))
        download_tts_btn = gr.Button(i18n("Baixar Áudio TTS"))
        download_tts_out = gr.File(label=i18n("Baixar WAV"))

        def serve_tts():
            tts_path = "/app/assets/audios/tts_output.wav"
            if not os.path.isfile(tts_path):
                raise gr.Error(i18n("Arquivo TTS não encontrado!"))
            return tts_path, "tts_output.wav"

        download_tts_btn.click(fn=serve_tts, inputs=[], outputs=[download_tts_out])

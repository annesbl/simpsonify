import os
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")   # HuggingFace Token
LORA_PATH = os.getenv("LORA_PATH") # Pfad oder HF-Repo für LoRA, z.B. "lora/simpson"



# Device wählen
if torch.backends.mps.is_available():
    DEVICE = "mps"   # Apple Silicon
elif torch.cuda.is_available():
    DEVICE = "cuda"  # NVIDIA GPU
else:
    DEVICE = "cpu"   # CPU fallback

print(f"Using device: {DEVICE}")
print(f"LoRA path: {LORA_PATH}")


def load_pipeline() -> StableDiffusionImg2ImgPipeline:
    if HF_TOKEN is None:
        raise RuntimeError(
        "HF_TOKEN ist nicht gesetzt. Bitte HuggingFace Token als Umgebungsvariable exportieren."
        )

    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        safety_checker=None,  # für Projekt okay, aber aufpassen was generiert wird
    )

    pipe = pipe.to(DEVICE)
    if DEVICE != "cpu":
        pipe.enable_attention_slicing()
    
    if LORA_PATH:
        print(f"Lade LoRA-Gewichte von: {LORA_PATH}")
        # Einfacher Fall: im Ordner liegt eine .safetensors-Datei
        pipe.load_lora_weights(LORA_PATH)
        # Optional: LoRA einbacken (dann später keine scale-Steuerung mehr nötig)
        # pipe.fuse_lora()
    else:
        print("Kein LORA_PATH gesetzt – Stable Diffusion läuft ohne LoRA.")

    return pipe


PIPE = load_pipeline()

app = FastAPI(title="Simpsonify Yourself API (Stable Diffusion + LoRA)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # für Demo/Dev ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Simpsonify Yourself API (Stable Diffusion + LoRA) is running 🚀"}

def prepare_image(input_path: Path, max_size: int = 768) -> Image.Image:
    """Lädt das Bild und skaliert es auf eine sinnvolle Größe für SD."""
    img = Image.open(input_path).convert("RGB")

    img.thumbnail((max_size, max_size), Image.LANCZOS)

    w, h = img.size
    w = w - (w % 8)
    h = h - (h % 8)
    if w == 0 or h == 0:
        raise ValueError("Bild ist zu klein.")
    img = img.resize((w, h), Image.LANCZOS)
    return img


def simpsonify_with_sd(
    input_path: Path,
    output_path: Path,
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    prompt_extra: Optional[str] = None,
    lora_scale: Optional[float] = None,
) -> None:
    """
    Führt ein Stable-Diffusion-Img2Img Sampling im Cartoon-/Simpson-Style durch.
    Nutzt optional eine LoRA mit einstellbarem lora_scale.
    """

    init_image = prepare_image(input_path)

    base_prompt = (
        "portrait of a person as a yellow-skinned 2D cartoon character, "
        "thick black outlines, simple flat shading, big expressive eyes, "
        "clean line art, vibrant colors, tv animation style"
    )
    if prompt_extra:
        prompt = base_prompt + ", " + prompt_extra
    else:
        prompt = base_prompt

    negative_prompt = (
        "deformed, extra limbs, blurry, low quality, distorted face, "
        "mutated hands, text, watermark, logo, ugly, glitch"
    )

    ca_kwargs = None
    if lora_scale is not None and LORA_PATH:
        ca_kwargs = {"scale": float(lora_scale)}

    autocast_device = DEVICE if DEVICE != "cpu" else "cpu"
    with torch.autocast(autocast_device):
        result = PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=30,
            cross_attention_kwargs=ca_kwargs,
        )

    out_image = result.images[0]
    out_image.save(output_path)


# -------------------------------------------------------
# API Endpoint: /api/simpsonify
# -------------------------------------------------------

@app.post("/api/simpsonify")
async def simpsonify_endpoint(
    file: UploadFile = File(...),
    strength: float = Form(0.6),
    guidance_scale: float = Form(7.5),
    extra_prompt: str = Form(""),
    lora_scale: float = Form(0.8),
):
    try:
        image_id = uuid.uuid4().hex
        input_path = UPLOAD_DIR / f"{image_id}_{file.filename}"
        output_path = OUTPUT_DIR / f"{image_id}.png"

        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        simpsonify_with_sd(
            input_path=input_path,
            output_path=output_path,
            strength=strength,
            guidance_scale=guidance_scale,
            prompt_extra=extra_prompt if extra_prompt.strip() else None,
            lora_scale=lora_scale,
        )

        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


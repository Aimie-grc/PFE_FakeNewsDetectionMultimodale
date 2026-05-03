from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


@dataclass
class QwenFaceConfig:
    model_name: str = "Qwen2-VL-2B"
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    family: str = "qwen2vl"
    max_new_tokens: int = 400
    temperature: float = 0.1
    resize_ratio: float = 0.5
    question: str = (
        "Analyse l'image et reponds uniquement avec un JSON valide, sans aucun texte avant/apres. "
        "Format exact attendu: "
        "{"
        "\"num_people_visible\": <int>, "
        "\"people\": ["
        "{\"sees_face\": <bool>, \"can_identify_person\": <bool>, \"person_name\": <string>, \"confidence\": <int 0-100>}"
        "], "
        "\"confidence\": <int 0-100>, "
        "\"justification\": <string courte>"
        "}. "
        "Regles: "
        "1) Si aucune personne visible: num_people_visible=0 et people=[]. "
        "2) Si can_identify_person=false alors person_name=\"\". "
        "3) Confidence doit etre un entier entre 0 et 100."
    )


class QwenFaceService:
    def __init__(self, config: QwenFaceConfig | None = None) -> None:
        self.config = config or QwenFaceConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self._hf_ner = None
        self._spacy_nlp = None

    # -----------------------------
    # Model loading / inference
    # -----------------------------
    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        if self.config.family != "qwen2vl":
            raise ValueError(f"Unsupported family: {self.config.family}")

        from transformers import Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

    @staticmethod
    def _to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in inputs.items():
            out[k] = v.to(device) if hasattr(v, "to") else v
        return out

    @staticmethod
    def _extract_assistant_text(decoded: str) -> str:
        txt = decoded.strip()
        if "assistant" in txt.lower():
            txt = txt.split("assistant")[-1].strip()
        return txt

    def resize_for_inference(self, image: Image.Image) -> Image.Image:
        ratio = max(0.01, min(1.0, self.config.resize_ratio))
        new_w = max(1, int(image.width * ratio))
        new_h = max(1, int(image.height * ratio))
        resample = (
            Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling")
            else Image.LANCZOS
        )
        return image.resize((new_w, new_h), resample=resample)

    def run_inference(self, image: Image.Image) -> str:
        self.load()
        assert self.model is not None and self.processor is not None

        self.model.eval()
        do_sample = self.config.temperature > 0
        temperature = self.config.temperature if do_sample else None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.config.question},
                ],
            }
        ]

        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(
                images=image, text=self.config.question, return_tensors="pt"
            )

        inputs = self._to_device(inputs, self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        if "input_ids" in inputs:
            new_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        else:
            new_ids = output_ids

        decoded = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return self._extract_assistant_text(decoded)

    # -----------------------------
    # Scoring helpers
    # -----------------------------
    @staticmethod
    def _strip_accents(s: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
        )

    def clean_person_candidate(self, name: str) -> str:
        name = self._strip_accents(name or "")
        name = re.sub(r"[^A-Za-z\s'-]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        noise_tokens = {"Chan", "Det", "Detective", "By", "PM", "UTC", "Sign", "Subscribe"}
        parts = [p for p in name.split() if p not in noise_tokens]
        return " ".join(parts).strip()

    def normalize_name(self, name: str) -> str:
        cleaned = self.clean_person_candidate(name)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    @staticmethod
    def parse_vlm_json(raw_output: str) -> tuple[dict[str, Any], str | None]:
        raw = (raw_output or "").strip()
        if not raw:
            return {}, "Sortie VLM vide"

        candidates = [raw]
        md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
        if md_match:
            candidates.append(md_match.group(1).strip())

        obj_match = re.search(r"\{.*\}", raw, flags=re.S)
        if obj_match:
            candidates.append(obj_match.group(0).strip())

        errors: list[str] = []
        seen: set[str] = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    return data, None
                errors.append("Le JSON n'est pas un objet")
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))

        err = " | ".join(errors[:3]) if errors else "Echec de parsing JSON"
        return {}, err

    def extract_people_from_vlm(self, vlm_data: dict[str, Any]) -> dict[str, str]:
            people = vlm_data.get("people", []) if isinstance(vlm_data, dict) else []
            detected: dict[str, str] = {}
            for p in people:
                if not isinstance(p, dict):
                    continue
                can_identify = p.get("can_identify_person") is True
                person_name = (p.get("person_name") or "").strip()
                if can_identify and person_name:
                    k = self.normalize_name(person_name)
                    if k:
                        detected[k] = person_name
            return detected

    def _load_spacy(self):
        if self._spacy_nlp is not None:
            return self._spacy_nlp
        try:
            import spacy

            self._spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:  # noqa: BLE001
            self._spacy_nlp = None
        return self._spacy_nlp

    def _load_hf_ner(self):
        if self._hf_ner is not None:
            return self._hf_ner
        try:
            from transformers import pipeline

            self._hf_ner = pipeline(
                "ner", model="dslim/bert-base-NER", aggregation_strategy="simple"
            )
        except Exception:  # noqa: BLE001
            self._hf_ner = None
        return self._hf_ner

    def extract_people_from_text(self, text: str) -> dict[str, str]:
        text = (text or "")
        if not text.strip():
            return {}

        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\b([A-Z][A-Za-z'-]+)'s\b", r"\1", cleaned)

        names: list[str] = []

        nlp = self._load_spacy()
        if nlp is not None:
            doc = nlp(cleaned)
            names.extend(ent.text for ent in doc.ents if ent.label_ == "PERSON")

        if not names:
            ner = self._load_hf_ner()
            if ner is not None:
                ents = ner(cleaned)
                for ent in ents:
                    label = ent.get("entity_group") or ent.get("entity")
                    if label in {"PER", "PERSON"}:
                        names.append(ent.get("word", ""))

        if not names:
            pattern = r"\b([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3})\b"
            names = re.findall(pattern, cleaned)

        extracted: dict[str, str] = {}
        for raw_name in names:
            cand = self.clean_person_candidate(raw_name)
            cand = re.sub(r"\s+", " ", cand).strip(" -")
            if len(cand.split()) < 2:
                continue
            key = self.normalize_name(cand)
            if not key or len(key) < 3:
                continue
            prev = extracted.get(key)
            if prev is None or len(cand) > len(prev):
                extracted[key] = cand
        return extracted

    @staticmethod
    def fold_aliases(source: dict[str, str], reference: dict[str, str]) -> dict[str, str]:
        if not source:
            return {}
        merged = dict(source)
        for s_key in list(source.keys()):
            s_tokens = set(s_key.split())
            if len(s_tokens) < 2:
                continue
            for r_key in reference.keys():
                r_tokens = set(r_key.split())
                if len(r_tokens) < 2:
                    continue
                if r_tokens.issubset(s_tokens):
                    merged.pop(s_key, None)
                    merged[r_key] = reference[r_key]
                    break
        return merged

    def build_coherence_result(self, article_text: str, vlm_output_text: str) -> dict[str, Any]:
        vlm_data, parse_error = self.parse_vlm_json(vlm_output_text)
        in_image = self.extract_people_from_vlm(vlm_data)
        in_text = self.extract_people_from_text(article_text)
        in_text = self.fold_aliases(in_text, in_image)

        all_keys = set(in_image.keys()) | set(in_text.keys())
        personnes: dict[str, str] = {}
        common = 0
        for k in sorted(all_keys):
            if k in in_image and k in in_text:
                status = "image et texte"
                display_name = in_text.get(k) or in_image.get(k)
                common += 1
            elif k in in_text:
                status = "texte"
                display_name = in_text[k]
            else:
                status = "image"
                display_name = in_image[k]
            personnes[display_name] = status

        total_text_people = len(in_text)
        score = (common / total_text_people) if total_text_people > 0 else 0.0
        face_score = 0 if score > 0 else 1

        return {
            "personnes": personnes,
            "score": score,
            "score_percent": round(score * 100, 2),
            "face_score": face_score,
            "parse_error": parse_error,
            "details": {
                "nb_personnes_texte": total_text_people,
                "nb_personnes_communes_image_texte": common,
                "personnes_texte": list(in_text.values()),
                "personnes_image": list(in_image.values()),
            },
            "raw_vlm": vlm_data,
        }

    def predict(self, image: Image.Image, article_text: str) -> dict[str, Any]:
        image = image.convert("RGB")
        inference_image = self.resize_for_inference(image)
        output_text = self.run_inference(inference_image)
        result = self.build_coherence_result(article_text, output_text)
        result["inference_image_size"] = inference_image.size
        result["model"] = {
            "name": self.config.model_name,
            "id": self.config.model_id,
            "device": self.device,
        }
        return result


# Singleton optionnel pour usage API simple
_SERVICE: QwenFaceService | None = None


def get_service() -> QwenFaceService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = QwenFaceService()
    return _SERVICE


"""Generate answers and extract visual features with a frozen Qwen model."""
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from earthjev.prompts import evidence_question

Image.MAX_IMAGE_PIXELS = None  # UHR scenes (LRS-VQA tifs reach 27k x 27k px) trip PIL's decompression guard

FACTOR = 32  # patch 16 x merge 2: one visual token per 32 x 32 px
# The processor rejects > 200:1. A 150:1 strip can exceed that once its short side is rounded to 32 px
# (7050x47 -> 7040x32 = 220:1); 100:1 cannot, since rounding shrinks a side by less than 2x.
MAX_ASPECT = 100
MIN_PIXELS = 65_536  # the processor's own floor (preprocessor_config.json shortest_edge)
DECISION_PIXELS = 256 * FACTOR * FACTOR  # decision views: at most 256 visual tokens (Sec. III-B, III-C)
LETTERS = 'ABCDE'


def _pad_extreme_aspect(image):
    """Pad a strip beyond MAX_ASPECT:1 with gray (128) to MAX_ASPECT:1."""
    w, h = image.size
    if max(w, h) / max(1, min(w, h)) <= MAX_ASPECT:
        return image
    if w >= h:
        nh = -(-w // MAX_ASPECT)
        canvas = Image.new('RGB', (w, nh), (128, 128, 128))
        canvas.paste(image, (0, (nh - h) // 2))
    else:
        nw = -(-h // MAX_ASPECT)
        canvas = Image.new('RGB', (nw, h), (128, 128, 128))
        canvas.paste(image, ((nw - w) // 2, 0))
    return canvas


def prepare_image(image, max_pixels, min_pixels=None):
    """PIL image -> RGB with both sides multiples of 32 and min_pixels <= w*h <= max_pixels (min_pixels None: the
    processor floor MIN_PIXELS): the size the processor's own smart_resize picks, reached with LANCZOS
    (reducing_gap 1, i.e. an integer box reduction first)."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = _pad_extreme_aspect(image)
    w, h = image.size
    th, tw = smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS if min_pixels is None else min_pixels,
                          max_pixels=max_pixels)
    if (tw, th) != (w, h):
        image = image.resize((tw, th), Image.LANCZOS, reducing_gap=1.0)
    return image


def chat_inputs(processor, images, text, system=None):
    """Processor inputs (CPU tensors) of one user turn: the images [(PIL image, max_pixels)], each prepared to its
    budget, then the text; with an optional system turn before it; thinking off."""
    min_pixels = processor.image_processor.size['shortest_edge']
    images = [prepare_image(im, px, min_pixels) for im, px in images]
    messages = [] if system is None else [{'role': 'system', 'content': [{'type': 'text', 'text': system}]}]
    messages.append({'role': 'user', 'content': [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': text}]})
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    kwargs = {}
    if images:  # prepared sizes are fixed points of the processor's smart_resize for any [min, max] around them
        areas = [im.width * im.height for im in images]
        kwargs = dict(images=images, images_kwargs=dict(min_pixels=min(areas), max_pixels=max(areas)))
    return processor(text=[prompt], return_tensors='pt', **kwargs)


def letter_token_ids(tokenizer):
    """Per letter A-E, the sorted first tokens of its single-token spellings 'C', ' C', '(C', '\\tC', ' (C'."""
    prefixes = ('', ' ', '(', '\t', ' (')
    return [sorted({e[0] for e in (tokenizer.encode(p + L, add_special_tokens=False) for p in prefixes) if len(e) == 1})
            for L in LETTERS]


class Answerer:
    """The frozen answering model (loaded once on the GPU). max_pixels: the default image budget."""

    def __init__(self, model_dir, dtype='bfloat16', attn_implementation='flash_attention_2', max_pixels=1_048_576):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype,
            attn_implementation=attn_implementation, device_map='cuda').eval()
        self.tokenizer = self.processor.tokenizer
        self.max_pixels = max_pixels
        ids = [self.tokenizer.encode(w, add_special_tokens=False) for w in ('Yes', 'No')]
        assert all(len(i) == 1 for i in ids), f'Yes/No must be single tokens, got {ids}'
        (self.yes_id,), (self.no_id,) = ids
        self.im_end = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.letter_token_ids = letter_token_ids(self.tokenizer)

    @torch.no_grad()
    def generate(self, images, text, max_new_tokens, *, system):
        """Greedy answer to the images [(PIL image, max_pixels)] and the text under a system turn, with
        tok_lp: the log-probs of the generated tokens up to and including the first <|im_end|> (all if none), and
        letter_lp: the log-prob of each letter A-E at the first position (logsumexp over its letter_token_ids).
        Generation stops at the model's end-of-sequence tokens (its generation configuration: <|endoftext|> for
        Qwen3.5-9B, which goes on past <|im_end|>; <|im_end|> or <|endoftext|> for Qwen3-VL-8B) or after
        max_new_tokens; the output text drops special tokens.
        -> dict(output, tok_lp, letter_lp, ended: an <|im_end|> was generated)."""
        inputs = chat_inputs(self.processor, images, text, system).to(self.model.device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, output_logits=True,
                                  return_dict_in_generate=True)
        new = out.sequences[0, inputs['input_ids'].shape[1]:].tolist()
        n = new.index(self.im_end) + 1 if self.im_end in new else len(new)
        lps = [torch.log_softmax(x[0].float(), -1) for x in out.logits[:n]]
        return dict(output=self.tokenizer.decode(new, skip_special_tokens=True).strip(),
                    tok_lp=[float(lp[t]) for lp, t in zip(lps, new)],
                    letter_lp=[float(lps[0][ids].logsumexp(0)) for ids in self.letter_token_ids],
                    ended=self.im_end in new)

    @torch.no_grad()
    def yes_probability(self, image, query):
        """y = p(Yes) of the [Yes, No] next-token logits for the evidence question about `query` on the 256-token
        view of `image` (the pointer-trust signal y_j, Sec. III-B), without a system turn."""
        view = prepare_image(image, DECISION_PIXELS, MIN_PIXELS)
        inputs = chat_inputs(self.processor, [(view, self.max_pixels)], evidence_question(query)).to(self.model.device)
        logits = self.model(**inputs, logits_to_keep=1).logits[0, -1, [self.yes_id, self.no_id]]
        confidence = 2 * (torch.softmax(logits.float(), dim=-1)[0].item() - 0.5)  # ZoomSearch's confidence in [-1, 1]
        return (confidence + 1) / 2

import jieba
import re
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics import CHRF, BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from comet import load_from_checkpoint
from typing import Dict, List


def lazy_singleton(func):
    _instance = None
    def wrapper(*args, **kwargs):
        nonlocal _instance
        if _instance is None:
            _instance = func(*args, **kwargs)
        return _instance
    
    return wrapper

@lazy_singleton
def get_emb_model():
    print("Loading embedding model...")
    return SentenceTransformer(
        "./bge-large-zh-v1.5"
        # "./bge-large-en-v1.5"
    )

@lazy_singleton
def get_xcomet_model():
    print("Loading xcomet model...")
    return load_from_checkpoint(
        "./XCOMET-XL/checkpoints/model.ckpt",
        reload_hparams=True, local_files_only=True,
    )


def get_xcomet_score(data: List[Dict[str, str]]):
    import torch
    gpu_count = torch.cuda.device_count()
    gpu_count = 1
    print(f'Xcomet Evaluation starting... (GPU={gpu_count})')
    xcomet_model = get_xcomet_model()
    model_output = xcomet_model.predict(data, gpus=gpu_count, 
                                        batch_size=32, num_workers=64,)
    print('Xcomet Evaluation finished...')
    return model_output.scores, model_output.metadata.error_spans


def get_embedding_score(references, hypothesis):
    emb_model = get_emb_model()
    embeddings_1 = emb_model.encode(references, normalize_embeddings=True, show_progress_bar=False)
    embeddings_2 = emb_model.encode(hypothesis, normalize_embeddings=True, show_progress_bar=False)
    cosine_similarity = embeddings_1 @ embeddings_2.T
    return cosine_similarity


def tokenize(text, lang='zh'):
    if lang == 'en':
        return word_tokenize(text)
    else:
        return list(jieba.cut(text))


def cal_bleu(reference_sentences, candidate_sentence, lang='zh'):
    references = [tokenize(ref, lang) for ref in reference_sentences]
    candidate = tokenize(candidate_sentence, lang)
    smooth_func = SmoothingFunction().method1
    bleu_score = sentence_bleu(references, candidate, smoothing_function=smooth_func)
    return bleu_score

def cal_sacre_blue(reference, hypothesis, lang='zh'):
    bleu = BLEU(tokenize=lang, effective_order=True,)
    sentence_score = bleu.sentence_score(hypothesis, [reference])
    sentence_score = sentence_score.score / 100
    return sentence_score

def cal_chrf(reference, hypothesis):
    chrf = CHRF(beta=2)
    score = chrf.sentence_score(hypothesis, [reference])
    score = score.score / 100
    return score


def cal_rouge(reference, hypothesis, lang='zh'):
    from rouge_score import rouge_scorer, tokenizers
    import jieba
    import re
    
    class ChineseTokenizer(tokenizers.Tokenizer):
        def tokenize(self, text):
            if re.search('[\u4e00-\u9fff]', text):
                return list(jieba.cut(text))
    
    if lang == "zh":
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=ChineseTokenizer())
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    try:
        score = scorer.score(reference, hypothesis)
        return score['rougeL'].fmeasure
    except Exception as e:
        print(f"Error calculating ROUGE-L: {e}")
        print(f"Reference: {reference[:50]}...")
        print(f"Hypothesis: {hypothesis[:50]}...")
        return 0.0


def get_bleu_chrf_bge_score(reference, candidate_sentence):
    chrf_score = cal_chrf(reference=reference, hypothesis=candidate_sentence)
    bleu_score = cal_bleu([reference], candidate_sentence)
    bge_score = get_embedding_score(reference, candidate_sentence)
    return bleu_score, chrf_score, bge_score


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))
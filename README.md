# RedTrans

**RedTrans** is a large-scale translation model tailored for social media platforms, designed to handle informal and culturally embedded expressions like slang, emojis, and memes in both Chinese and English.

---

### 🔍 Overview

Traditional translation systems often struggle with the highly contextual and culturally nuanced content found on social media—think memes, pop culture references, internet slang, and emojis. RedTrans bridges this gap by combining large language models with fine-tuning on social-domain data and preference-aligned training to deliver more accurate and culturally aware translations.

---

### ✨ Key Features

- 🧠 **Social-context aware translation:** Designed to handle slang, internet memes, emoji semantics, and culturally rich expressions  
- 📊 **RedTrans-Bench benchmark:** 2,800+ curated translation cases from real-world social media content, including posts, comments, and captions  
- 🔁 **Dual-LLM back-translation sampling:** Enhances data diversity and improves generalization  
- 🎯 **RePO (Rewritten Preference Optimization):** Human-in-the-loop preference correction for cleaner RLHF signal  
- ✅ **Production-ready:** Already deployed in a real-world social app used by millions

---

### 📦 Project Structure

```bash
.
├── Data License              # License for dataset usage
├── README.md                 # Project documentation (this file)
├── RedTrans-Bench.json       # Benchmark dataset for SNS translation
└── utils.py                  # Utility functions (data processing, evaluation, etc.)
```

---

### 📊 Performance

RedTrans achieves strong results across public MT benchmarks, and significantly outperforms general-purpose models on the RedTrans-Bench evaluation set.

| Benchmark         | BLEU ↑ | chrF++ ↑ |
|------------------|--------|-----------|
| WMT22–24         | ✅     | ✅         |
| FLORES200        | ✅     | ✅         |
| **RedTrans-Bench** | 🥇     | 🥇         |

---

### 🛠️ Usage

> 📌 Inference scripts and training code will be released soon. Stay tuned for updates!

---

### 📚 Citation

If you find this project helpful, please consider citing.

---

### 🤝 Contributing & Contact

Feel free to open an issue or submit a PR. If you’re interested in NLP for social media, we’d love to collaborate or hear from you!

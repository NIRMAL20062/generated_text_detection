# Error analysis: what the detectors get wrong (issue #27)

Two cases: **RoBERTa on `logo_bloomz`** (BLOOMZ never seen in training, the hardest held-out generator) and
**TF-IDF + LR on `indist`** (the near-perfect supervised baseline). For each, the 5 most confident false positives
(human text the detector was surest was AI) and false negatives (AI text it was surest was human) on the **test**
set, at the threshold `src.eval` chose on validation. The examples are not hand-picked; regenerate them with

```bash
python -m src.error_analysis --method roberta  --setting logo_bloomz   # -> results/roberta/logo_bloomz/errors.md
python -m src.error_analysis --method tfidf_lr --setting indist        # -> results/tfidf_lr/indist/errors.md
```

The aggregate numbers (error rate per domain, median lengths) come from the same `preds.csv` + `load_texts()`.
"Why" is our reading of the text, a hypothesis, not something the model told us. Human texts have no generator of
their own, so that column is "–" for them.

## 1. RoBERTa, BLOOMZ held out (`logo_bloomz`)

AUROC 0.881, threshold 0.997. Test errors: **193 of 5,162 human texts** called machine (3.7%), **2,863 of 5,000
BLOOMZ texts** called human (57%).

### False positives (human called AI)

| id | domain | score | chars | first ~200 characters | probable reason |
|---|---|---|---|---|---|
| `wikihow_bloomz_1873_h` | wikihow | 0.9992 | 9,356 | Begin by applying a light brown, matte eyeshadow to the crease of your eyelid. Use a tapered brush to apply the eyeshadow all along the crease. Create a light layer of eyeshadow. This part of your smo… | Imperative, repetitive "how-to" steps; very long, and RoBERTa only sees the first 256 tokens, which read like a generated tutorial. |
| `arxiv_dolly_1061_h` | arxiv | 0.9991 | 1,439 | Community structure is an important property of complex networks. An automatic discovery of such structure is a fundamental task in many disciplines, including sociology, biology, engineering, and com… | Generic, textbook opening sentence ("X is an important property of Y") is exactly what generators write for abstracts. |
| `peerread_chatGPT_0163_h1` | peerread | 0.9990 | 251 | The paper provides a detailed analysis of the instability issues surrounding the training of GANs. They demonstrate how perturbations can help with improving stability. Given the popularity of GANs, t… | Very short, polite, summary-style review with no specific criticism: the template of machine reviews. |
| `wikipedia_chatGPT_0862_h` | wikipedia | 0.9990 | 1,662 | Seomun Market is the largest traditional street market in Daegu, South Korea, containing more than 4,000 shops. Seomun Market is particularly known as a source for textiles and sewing services, a key … | Clean encyclopedic lead with no page leftovers (no "references"/"external links"), so it looks like a generated article. |
| `arxiv_cohere_1619_h` | arxiv | 0.9990 | 1,224 | We present a method for the direct and continuous separation of red and white blood cells from plasma at the microscale. The method is implemented in a microfluidic system with magnetic functionality. … | Formulaic "We present a method…" abstract; formal scientific style overlaps with machine abstracts. |

### False negatives (AI called human)

| id | domain | generator | score | chars | first ~200 characters | probable reason |
|---|---|---|---|---|---|---|
| `wikihow_bloomz_1743_m` | wikihow | bloomz | 0.0013 | 2,415 | Removing or installing a car door can be done at home, but requires some knowledge about how cars operate. This guide assumes no prior experience working on automobiles; however if you’re familiar wit… | Chatty, first-person-guide tone with contractions and curly quotes, unlike the polished style of the 4 training generators. |
| `wikihow_bloomz_1507_m` | wikihow | bloomz | 0.0013 | 979 | Infection during chemotherapy occurs because the drugs weaken your immune system. The risk increases if you've had chemo before; have other health problems that affect your immunity, including diabete… | Plain, direct register with semicolons and lists, closer to human wikiHow writing than to ChatGPT/davinci style. |
| `wikihow_bloomz_0138_m` | wikihow | bloomz | 0.0014 | 1,106 | Read psychology journals like any other academic text. Start at the beginning! If possible start reading your chosen journal issue before class starts so you are prepared when it comes time to discuss… | Exclamations and missing commas ("If possible start…"): informal, slightly ungrammatical, which RoBERTa learned to call human. |
| `wikihow_bloomz_2146_m` | wikihow | bloomz | 0.0014 | 1,297 | If you are trying to teach your child how to use the toilet but find yourself struggling because it's not working then try these tips. If you're having trouble getting your little one interested in us… | Run-on sentences and colloquial phrasing ("your little one"), again a human-like register. |
| `wikihow_bloomz_0534_m` | wikihow | bloomz | 0.0015 | 1,216 | Work abroad offers many benefits. You’ll get to live somewhere new while gaining professional... Read more » , which provides information on how to find positions at embassies around the world. , whic… | BLOOMZ copied web-page debris ("Read more »", dangling ", which"), the kind of leftover RoBERTa associates with human scraped text. |

## 2. TF-IDF + LR, in distribution (`indist`)

AUROC 0.990, threshold 0.467. Test errors: **105 of 1,339 human texts** called machine (7.8%), **80 of 2,499 machine
texts** called human (3.2%).

### False positives (human called AI)

| id | domain | score | chars | first ~200 characters | probable reason |
|---|---|---|---|---|---|
| `arxiv_davinci_1854_h` | arxiv | 0.986 | 1,301 | (Abridged) The discovery of extrasolar planetary systems revealed an unexpected diversity of planetary systems that has revolutionized planet formation theory. A strong program of theoretical research… | Broad, review-like abstract with words ("revealed", "diversity", "revolutionized") that are strong machine n-grams. |
| `arxiv_chatGPT_0424_h` | arxiv | 0.977 | 1,286 | A library of C functions yielding exact solutions of potential and flux influences due to uniform surface distribution of singularities on flat triangular and rectangular elements has been developed. … | Formal passive voice and no LaTeX or citations, so none of the "human" formatting features fire. |
| `arxiv_bloomz_1833_h` | arxiv | 0.971 | 1,254 | Security of information has become a major issue during the last decades. New algorithms based on chaotic maps were suggested for protection of different types of multimedia data, especially digital i… | Generic opening ("has become a major issue") that generators also use to start abstracts. |
| `peerread_cohere_0273_h9` | peerread | 0.963 | 526 | Table 2 is missing some recent results on this task. Please see the NTI and NSE results on the same task [1,2]. NTI is particularly relevant to this work because it encodes a sentence with an n-ary tr… | Short, clean review; review vocabulary ("results", "task", "this work") is shared with machine reviews. |
| `arxiv_chatGPT_1240_h` | arxiv | 0.954 | 1,340 | Performance of object-oriented database systems (OODBs) is still an issue to both designers and users nowadays. The aim of this paper is to propose a generic discrete-event random simulation model, ca… | "The aim of this paper is to propose" is a machine-like phrase; formal, generic abstract. |

### False negatives (AI called human)

| id | domain | generator | score | chars | first ~200 characters | probable reason |
|---|---|---|---|---|---|---|
| `wikipedia_bloomz_0783_m` | wikipedia | bloomz | 0.001 | 186 | \* Main Page - The official website of the Belgian filmmaker Florian Verhaeghe \* Filmography - List of films directed by Florian Verhaeghe (in French) \* External links - Official websites | A list of links containing "External links", one of the strongest *human* TF-IDF features (a Wikipedia page leftover); very short. |
| `arxiv_bloomz_1527_m` | arxiv | bloomz | 0.027 | 1,082 | We study the deformation of the N=4 supersymmetric gauge theories in two dimensions by real and complex parameters $\beta$, which are related to the coupling constant $\lambda$ as $\lambda = \beta^2$ … | LaTeX math (`$\beta$`) appears almost only in human arXiv abstracts, so it pulls the score to human. |
| `arxiv_davinci_0875_m` | arxiv | davinci | 0.038 | 840 | A comprehensive study into the degree of ionization of the pre-star state Barnard 68 is undertaken by observing and modelling millimetric H13CO+ and DCO+ lines. Comparison to existing CO isotopologue … | Dense technical terms (H13CO+, isotopologue) that occur in human abstracts, few generic machine phrases. |
| `peerread_dolly_0496_m2` | peerread | dolly | 0.055 | 2,123 | Overall, this paper makes a strong case for developing noise models for Latent Space Modeling and provides empirical validation of such a noise model on a variety of tasks. I especially like the notio… | First-person opinion ("I especially like") is typical of human reviewers. |
| `arxiv_dolly_0394_m` | arxiv | dolly | 0.057 | 2,809 | A photometric study of stellar structures throughout the disk of the barred galaxy NGC 3367 is presented. This is a nearby (distance of 23.9 ± 0.7 megaparsecs, or approximately 62 million light-years) … | Precise numbers with uncertainties (23.9 ± 0.7) and object catalogue names, a human-abstract signal. |

## Patterns

1. **On an unseen generator, the threshold is the problem more than the ranking.** RoBERTa still ranks BLOOMZ text
   above human text fairly well (AUROC 0.881), but the threshold chosen on the four training generators (0.997) is
   above the median BLOOMZ score (0.996), so 57% of BLOOMZ texts are missed while only 3.7% of humans are flagged.
   All five most confident misses are informal wikiHow texts: BLOOMZ writes in a looser, more human register than
   ChatGPT, davinci, Cohere and Dolly, so "informal = human" is the shortcut that fails.
2. **arXiv is where human texts get flagged.** TF-IDF flags 22% of human arXiv abstracts (57 of 260) against at most
   9% in any other domain, and arXiv also has RoBERTa's highest false-positive rate on `logo_bloomz` (8.2%). Formal,
   formulaic abstracts ("We present…", "The aim of this paper is…") look like generated text; flagged human texts
   are also shorter than correctly classified ones (median 1,166 vs 1,736 characters for TF-IDF).
3. **Formatting leftovers decide some cases.** The single most confident TF-IDF miss is a BLOOMZ text that happens
   to contain "External links", and LaTeX markup or precise numbers pull machine abstracts towards "human". This is
   the formatting-artifact limitation of the report in action: the model partly learned *where* human texts were
   scraped from, not *who* wrote them.
4. **Short, generic reviews are ambiguous.** Short PeerRead reviews without specific criticism are called machine,
   and machine reviews with first-person opinions are called human, so review length and specificity carry more
   signal than they should.

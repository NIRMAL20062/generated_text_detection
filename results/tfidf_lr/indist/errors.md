## tfidf_lr / indist

Threshold 0.4674 (chosen on val by `src.eval`; predict machine if score >= threshold). Test errors: 105 of 1339 human texts called machine (false positives), 80 of 2499 machine texts called human (false negatives).

### 5 most confident false positives (human text, highest scores)

| id | domain | generator | score | chars | first 200 characters |
|---|---|---|---|---|---|
| `arxiv_davinci_1854_h` | arxiv | – | 0.9862 | 1301 | (Abridged) The discovery of extrasolar planetary systems revealed an unexpected diversity of planetary systems that has revolutionized planet formation theory. A strong program of theoretical research |
| `arxiv_chatGPT_0424_h` | arxiv | – | 0.977 | 1286 | A library of C functions yielding exact solutions of potential and flux influences due to uniform surface distribution of singularities on flat triangular and rectangular elements has been developed.  |
| `arxiv_bloomz_1833_h` | arxiv | – | 0.9712 | 1254 | Security of information has become a major issue during the last decades. New algorithms based on chaotic maps were suggested for protection of different types of multimedia data, especially digital i |
| `peerread_cohere_0273_h9` | peerread | – | 0.9632 | 526 | Table 2 is missing some recent results on this task. Please see the NTI and NSE results on the same task [1,2]. NTI is particularly relevant to this work because it encodes a sentence with an n-ary tr |
| `arxiv_chatGPT_1240_h` | arxiv | – | 0.9541 | 1340 | Performance of object-oriented database systems (OODBs) is still an issue to both designers and users nowadays. The aim of this paper is to propose a generic discrete-event random simulation model, ca |

### 5 most confident false negatives (machine text, lowest scores)

| id | domain | generator | score | chars | first 200 characters |
|---|---|---|---|---|---|
| `wikipedia_bloomz_0783_m` | wikipedia | bloomz | 0.001349 | 186 | * Main Page - The official website of the Belgian filmmaker Florian Verhaeghe * Filmography - List of films directed by Florian Verhaeghe (in French) * External links - Official websites |
| `arxiv_bloomz_1527_m` | arxiv | bloomz | 0.02732 | 1082 | We study the deformation of the N=4 supersymmetric gauge theories in two dimensions by real and complex parameters $\\beta$, which are related to the coupling constant $\\lambda$ as $\\lambda = \\beta^2$  |
| `arxiv_davinci_0875_m` | arxiv | davinci | 0.03768 | 840 | A comprehensive study into the degree of ionization of the pre-star state Barnard 68 is undertaken by observing and modelling millimetric H13CO+ and DCO+ lines. Comparison to existing CO isotopologue  |
| `peerread_dolly_0496_m2` | peerread | dolly | 0.05476 | 2123 | Overall, this paper makes a strong case for developing noise models for Latent Space Modeling and provides empirical validation of such a noise model on a variety of tasks. I especially like the notio |
| `arxiv_dolly_0394_m` | arxiv | dolly | 0.05655 | 2809 | A photometric study of stellar structures throughout the disk of the barred galaxy NGC 3367 is presented. This is a nearby (distance of 23.9 ± 0.7 megaparsecs, or approximately 62 million light-years) |

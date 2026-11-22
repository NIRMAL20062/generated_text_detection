## roberta / logo_bloomz

Threshold 0.997 (chosen on val by `src.eval`; predict machine if score >= threshold). Test errors: 193 of 5162 human texts called machine (false positives), 2863 of 5000 machine texts called human (false negatives).

### 5 most confident false positives (human text, highest scores)

| id | domain | generator | score | chars | first 200 characters |
|---|---|---|---|---|---|
| `wikihow_bloomz_1873_h` | wikihow | – | 0.9992 | 9356 | Begin by applying a light brown, matte eyeshadow to the crease of your eyelid. Use a tapered brush to apply the eyeshadow all along the crease. Create a light layer of eyeshadow. This part of your smo |
| `arxiv_dolly_1061_h` | arxiv | – | 0.9991 | 1439 | Community structure is an important property of complex networks. An automatic discovery of such structure is a fundamental task in many disciplines, including sociology, biology, engineering, and com |
| `peerread_chatGPT_0163_h1` | peerread | – | 0.999 | 251 | The paper provides a detailed analysis of the instability issues surrounding the training of GANs. They demonstrate how perturbations can help with improving stability. Given the popularity of GANs, t |
| `wikipedia_chatGPT_0862_h` | wikipedia | – | 0.999 | 1662 | Seomun Market is the largest traditional street market in Daegu, South Korea, containing more than 4,000 shops. Seomun Market is particularly known as a source for textiles and sewing services, a key  |
| `arxiv_cohere_1619_h` | arxiv | – | 0.999 | 1224 | We present a method for the direct and continuous separation of red and white blood cells from plasma at the microscale. The method is implemented in a microfluidic system with magnetic functionality. |

### 5 most confident false negatives (machine text, lowest scores)

| id | domain | generator | score | chars | first 200 characters |
|---|---|---|---|---|---|
| `wikihow_bloomz_1743_m` | wikihow | bloomz | 0.001279 | 2415 | Removing or installing a car door can be done at home, but requires some knowledge about how cars operate. This guide assumes no prior experience working on automobiles; however if you’re familiar wit |
| `wikihow_bloomz_1507_m` | wikihow | bloomz | 0.001328 | 979 | Infection during chemotherapy occurs because the drugs weaken your immune system. The risk increases if you've had chemo before; have other health problems that affect your immunity, including diabete |
| `wikihow_bloomz_0138_m` | wikihow | bloomz | 0.001435 | 1106 | Read psychology journals like any other academic text. Start at the beginning! If possible start reading your chosen journal issue before class starts so you are prepared when it comes time to discuss |
| `wikihow_bloomz_2146_m` | wikihow | bloomz | 0.001449 | 1297 | If you are trying to teach your child how to use the toilet but find yourself struggling because it's not working then try these tips. If you're having trouble getting your little one interested in us |
| `wikihow_bloomz_0534_m` | wikihow | bloomz | 0.00149 | 1216 | Work abroad offers many benefits. You’ll get to live somewhere new while gaining professional... Read more » , which provides information on how to find positions at embassies around the world. , whic |

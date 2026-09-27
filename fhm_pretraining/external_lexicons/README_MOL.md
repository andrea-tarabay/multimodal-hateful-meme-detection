
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7787172.svg)](https://doi.org/10.5281/zenodo.7787173)


<h2 align="center"> MOL - Multilingual Offensive Lexicon Annotated with Contextual Information </h2>  

<p align="justify"> The MOL is the first specialized lexicon for hate speech detection annotated with contextual information. It consists of 1,000 explicit and implicit human-annotated rationales with pejorative connotations, manually identified by a linguist and annotated by three different annotators with contextual labels (context-dependent or context-independent). For example, the term "stupid" is classified as a context-independent offensive term, as it is predominantly used in pejorative contexts. In contrast, terms like "useless" and "worm" are considered context-dependent offensive terms because they can appear in both neutral and pejorative contexts. For instance: (i) non-pejorative, such as "this smartphone is useless" or "the fisherman uses worms for bait"; and (ii) pejorative, such as "this last president was useless" or "this human being is such a worm".</p>

The Multilingual Offensive Lexicon (MOL) was manually extracted by a linguist from the HateBR dataset, and each term and expression was annotated by three different annotators, achieving a high inter-annotator agreement score (73% Kappa). Originally written in Portuguese, MOL was manually translated by native speakers into English, Spanish, French, German, and Turkish, with cultural adaptations taken into account. Therefore, MOL is available in six different languages.

The table below describes the MOL statistics
<div align="center">
<table> 
<tr><th>Contextual Information</th><th>Hate Targets </th></tr>
<tr><td>

|class|label|total|
|--|--|--|  
|Context-independent offensive terms|1|612| 
|Context-dependent offensive terms|0|387| 
 |Total||1,000| 


</td><td>

|class|total|  
|--|--|  
|non-target|864|
|partyism|69|
|sexism|35|
|homophobia|16|
|fatphobia|9|
|religious intolerance|9|
|antisemitism|1|
|apology for the dictatorship|5|
|racism|4|  
|antisemitistm|3| 
|total|1,000|


</td></tr></table>
</div>

<h2 align="left"> CITING / BIBTEX </h2>

Please cite our paper if you use our lexicon:

```bibtex
 @article{Vargas_Carvalho_Pardo_Benevenuto_2024, 
 author={Vargas, Francielle and Carvalho, Isabelle and Pardo, Thiago A. S. and Benevenuto, Fabrício},
 title={Context-aware and expert data resources for Brazilian Portuguese hate speech detection}, 
 DOI={10.1017/nlp.2024.18}, 
 journal={Natural Language Processing},  
 year={2024}, 
 pages={435-456},
 volume{31},
 number={2},
 url={https://www.cambridge.org/core/journals/natural-language-processing/article/contextaware-and-expert-data-resources-for-brazilian-portuguese-hate-speech-detection/7D9019ED5471CD16E320EBED06A6E923#},
 }
```

<br>
<br>





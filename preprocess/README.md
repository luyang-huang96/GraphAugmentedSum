### process data    

1. How to extract triples  
```
java -mx32g -cp stanford-corenlp-3.8.0.jar:stanford-corenlp-3.8.0-models.jar:CoreNLP-to-HTML.xsl:slf4j-api.jar:slf4j-simple.jar \
    edu.stanford.nlp.naturalli.OpenIE \
    -threads 8 \
    -resolve_coref true \
    -ssplit.newlineIsSentenceBreak always \
    -format reverb \ 
    -filelist filelist.txt;
```    

2. You can obtain coreference resolution results from [stanfordnlp](https://stanfordnlp.github.io/CoreNLP/)

    
3. You can refer to following code to get constructed graph 
```
OpenIE_process.py
```
    

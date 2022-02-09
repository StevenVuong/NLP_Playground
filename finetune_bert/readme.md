[Original Post](https://medium.com/swlh/fine-tuning-bert-for-text-classification-and-question-answering-using-tensorflow-framework-4d09daeb3330#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjE4MmU0NTBhMzVhMjA4MWZhYTFkOWFlMWQyZDc1YTBmMjNkOTFkZjgiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2NDQyOTkwOTAsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExMjk2MzczMjI5MjcwMzU1ODc4OCIsImVtYWlsIjoic3RldmVudnVvbmc5NkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6IlN0ZXZlbiBWdW9uZyIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQVRYQUp3N0phNE5VU1hvOW9lMkpiRlp1cENMdkxsam8yRmFKRUxTOTlNWT1zOTYtYyIsImdpdmVuX25hbWUiOiJTdGV2ZW4iLCJmYW1pbHlfbmFtZSI6IlZ1b25nIiwiaWF0IjoxNjQ0Mjk5MzkwLCJleHAiOjE2NDQzMDI5OTAsImp0aSI6ImI3ODRmMDJkZmM5ZTNhMmRhZGU4YTNhYWQ2NGUwYjJkMzNjYzIwMmYifQ.hAbZE_wmOwt8TTBPQk-8lH1Ll6BVtLOmKo_QQHZ2Dg2Ha0vjzoYyfEFO-O0Pco57G8Exz_QViLvl8OkEkBHNo-3sGp2NhYxsxQW_zeACtv3Z063vve2AGw_zX2kG7TNYL4vS5kQ8QrLjgpf6sT4gR96f-07aC-yqC-KdDJgGwe-iKcR7ZrwGIpUNXLrxKAahp2jo7jRXQJPvKpqEFwmLCJHYMhj_imWmupR2zG9Y2j69O26s9ipRWa0w7_miMPIM2gqbScICGbpj3lai1A619xkSnjgIoM_mUhEZeCEupH8Bk_ZFBjzbWGfNjrWHVQyC9ASzDSQMjR8J7ZzJao9EOg)
# Fine Tuning BERT for Text Classification & QA with TensorFlow

Key Ideas:
-  Attention, without RNN's; more computationally attractive
-  Represent words as subgrams or ngrams (memory advantage)
-  Pre-trained BERT model can be used for wide variety of NLP tasks with fine-tuning (transfer learning)
    -  Can fine-tune all layers of pretrained transformer encoder; and additional output layer trained from scratch.
-  BERT word embeddings are context dependent, unlike Word2Vec or GloVe
-  Encodes context bidirectionally, while GPT only looks forward


BERT-Large cannot be trained on a regular consumer-grade GPU and must be sharded..<P>
BERT input sequence supports both single text and text-pairs. <br>
For single text, the input sequence is the concatenation of special classification token CLS, tokens of a text sequence and the special separation token SEP. <br>
In the latter; the BERT input is the concatenation of CLS, tokens of the first text sequence, SEP, tokens of the second text sequence, then SEP.
<P>
BERT model expects three inputs:

-  input ids: For classification problem, two input sentences should be tokenized and concatenated together.
-  input masks: Allow the model to cleanly differentiate between content and padding. Mask has same shape as input ids, and contains 1 anywhere the input ids is not padding.
-  input types: Same shape as input ids, but inside non-padded region, contains 0 or 1 indicating which sentence the token is a part of.

And the model returns two outputs: 

-  pooled output: Final hidden state corresponding to CLS token. It is used as the aggregate sequence representation for classification tasks. (roughly speaking, it is an embedding for the whole sentence.)
-  sequence output: 768 dimension embeddings for each token in the given sequence

Sparse categorical cross-entropy loss function used for both text classification and question answering tasks.
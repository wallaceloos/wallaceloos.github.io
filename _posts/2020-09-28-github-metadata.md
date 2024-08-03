---
layout: post
title:  "Transformer: GPT, I am your father!"
author: (Wallace S. Loos & Raphael F. Casseb)
date: 2024-07-08 21:01:00
description: A Transformer archicheture explanation focused on the attention mechanism
tags: NLP

# categories: Natural Language Processing
---

<p align="justify">Behind every great language large model (LLM) there is a transformer. Okay, maybe not every LLM model, but the transformer was a revolutionary architecture that changed everything and it is part directly or indirectly of our lives. However, before attacking the transformers (lol), let’s talk about the motivation behind it. After that we will study the transformer architecture and then take a look at GPT architecture.</p>

### The Translation Problem

<p align="justify">How do we translate one sentence from language A to language B? Many have tried to sort it out using technology, dating as far back as the Cold War. At that time, the first attempts were mostly rule-based systems akin to an automatic search in the dictionary. A very rudimentary (but exciting) solution.</p>

<p align="justify">Many decades later, we saw the era of <b>statistical machine translation</b> (1990-2010). Wise humans came up with the idea of splitting the translation task into two parts: the translation, per se, of words and small groups of words (like expressions or slang - again, like a fancy dictionary) + the combination of these “pieces” into meaningful texts (a writing/language model). These systems were incredibly effective but required a lot of human effort to maintain (e.g., it was necessary to create tables of correspondence between languages for the translation part, which had to be updated and maintained).</p>

<p align="justify">Then neural machine translation began its reign in the 2010s. Among the models developed, the seq2seq (or sequence-to-sequence) soon became the state of the art in the translation field. It was a fringe area of research in 2014, and by 2016 it had already become the leading standard method. Instead of a two-part model (translator + writer), it is based on a single neural network. The seq2seq architecture is composed of two recurrent neural networks (RNNs), one to encode the input sequence, and other to decode the output sequence. This encoder-decoder framework  - without the attention mechanism - was very prominent and widely used, but they presented a memory problem: they do quite well with short sentences but struggle with longer ones, even using Long Short-Term Memory (LSTM). They tend to “ignore” words at the beginning of long sentences, because all the information had to be encoded into a fixed length vector. This problem was known as the bottleneck problem. In addition,  RNNs architectures are “pricy” to use, (they are very computationally expensive) because you can only move forward with the next piece of data when the previous one is finished, which makes it difficult to parallelize the model.</p> 

<p align="justify">The attention mechanism is the secret weapon that balances the weight of ALL words, avoiding this “amnesia” and allowing the incorporation of large context (i.e., not forgetting distant words) into the translation task. That solves the bottleneck problem and helps with the interpretability of the model. There are multiple ways to compute the attention score: dot-product, multiplicative attention, or additive attention. An dot-product attention mechanism example is depicted below. The dot-product is taken from the encoder states and the decoder state followed by a softmax to turn the scores in a probability distribution. Then the weighted sum is computed from the decoder state and the probability distribution.</p> 

<p align="center">
<img src="/assets/img/seq2seq.png" width="45%" height="45%"> 
</p>
<p align="center"><font size="2">Image adapted from: [3]</font></p>

We must strongly highlight that the attention mechanism was an additional feature to the neural machine translation model. But it was such a dominating feature that the architecture in which it was present (the transformer) became a new reference in the field.

### Transformer

<p align="justify">A transformer is an architecture composed of an encoder, a decoder, and an attention mechanism. There are many adaptations and different implementations of transformers, but they stem from the common ancestor encoder + decoder + attention. The RNNs were removed and a stacked self-attention and point-wise, fully connected layers for both the encoder and decoder were added. This modification made it possible for the parallelization of the model since there is no time dependency. In this  section we will give more focus to the attention mechanism because (attention is all you need) it was what made transformer architecture so popular.</p> 

##### The Attention Mechanism

<p align="justify">The attention mechanism is a brilliant concept that allows the model to focus on different parts of the input sentence when producing each word of the output sentence. Imagine you are reading a long sentence and trying to translate it into another language. Naturally, you might pay more attention to certain words or phrases that are crucial for understanding the overall meaning. The attention mechanism mimics this human capability by assigning different weights to different words in the input sentence, effectively highlighting the most relevant ones for each step of the translation process improving the word alignment.</p>

<p align="justify">This mechanism works by creating a set of attention weights that measure the importance of each word in the input sentence relative to each word in the output sentence being generated. These weights are dynamically adjusted as the model processes the sentence, allowing it to consider the context provided by the entire input sequence. This dynamic adjustment helps the model maintain a better understanding of the context, leading to more accurate translations.</p> 

##### Key, Query, and Value

<p align="justify">At the heart of the attention mechanism are three concepts: key, query, and value. These are essentially linear transformations of the input data. The query represents the word currently being processed in the output sentence, the keys represent all words in the input sentence, and the values are the same as the keys but are used to generate the weighted sum. By calculating the dot product between the query and each key, the model determines the relevance of each word in the input sentence to the current word in the output sentence. This process generates a set of attention scores, which are then used to create a weighted sum of the values.
These linear transformations allow the model to incorporate information from the entire input sequence into each step of the translation process. This means that the model can dynamically adjust its focus, ensuring that even distant words in the input sentence can influence the translation of a given word in the output sentence. This sophisticated mechanism is what enables transformers to handle complex translation tasks with high accuracy and efficiency.</p> 

<p align="justify">Let’s use an example to explain how to compute an attention score of the sentence: “How are you?” First, the sentence is divided into a collection of tokens, this is called tokenization. After this, we have four different tokens: [How, are, you, ?], and our vocabulary is four, because we only have a single sentence with four tokens. The tokenization of the words is encoded to numbers: How = 11, are = 32, you = 12, ? = 3. In this example, we are doing word-based tokenization and randomly assigning numbers to the words without repeating them, but there are different methods of tokenization and encoding, such as Byte-Pair Encoding (BPE), the one used in GPT. The encoded words are then converted into vectors; this process is called embedding. Embedding represents data as vectors in D-dimensional space. This helps to better capture the meaning and relationships among the tokens. There is also a position embedding step that we are omitting here. In our example, D = 6. Each of the N tokens is represented by a vector of dimension 6. Because we have four tokens, we end up with a 4 x 6 matrix. Three different matrices Wq, Wk, and Wv (in practice, they are fully connected layers) are defined to compute the Query, Key, and Value. The attention score is the multiplication of the Query and Key scaled by a factor of the square root of d​. The softmax function is also applied after these operations to convert the result into a probability distribution. The output of the self-attention mechanism is the multiplication of the attention score and Value. The figure below describes all the steps mentioned previously.</p> 

<p align="center">
<img src="/assets/img/one_head.png" width="75%" height="75%"> 
</p>
<p align="center"><font size="2">Image adapted from: [17]</font></p>

<p align="justify">However, we need more attention, we need multi-head self-attention! Multi-head self-attention is the core element in a transformer network. It is the combination of many self-attention mechanisms and helps to focus on different aspects of the sequence. The figure below is an example of a multi-head self attention with 2 heads, where each head is represented by a self-attention mechanism. The total computational cost is equivalent to compute one single-head attention, because each head has its dimension reduced by D/H, where H is the number of heads.</p> 

<p align="center">
<img src="/assets/img/multi_head.png" width="75%" height="75%"> 
</p>
<p align="center"><font size="2">Image adapted from: [17]</font></p>

<p align="justify">The success of Transformer inspired the creation of other models, like GPT, which we are going to talk about in the next section.</p> 

### Generative Pre-trained Transformer (GPT)

<p align="justify">GPT is now a landmark in history. It is a groundbreaking model developed by OpenAI, that utilizes a decoder-only Transformer architecture and can be used on a wider spectrum of tasks. The model is conditioned not only on the input but also on the task P(output | input, task). The training process consists of two-phases: </p> 


* <p align="justify">Pre-training: the model is initially trained on a massive corpus of text data, allowing it to learn a ridiculously large amount of language patterns and structures. During this phase, GPT learns to predict the next word in a sentence, which helps it develop a deep understanding of language context and syntax. Now and then new pre-trained  versions of the model are released</p>

* <p align="justify">Fine-tuning: after pre-training, GPT can be fine-tuned on specific datasets for various language tasks, such as translation, summarization, and text generation. </p>


<p align="justify">Its ability to generate coherent and contextually relevant text has made it a powerful tool in many applications. Recently, GPT has also allowed users to create tailored tools based on GPT, that are personalized to do well in very specific tasks. You may have a GPT that is very good at writing code, another that is a specialist in finance and so on. The GPT architecture is presented in the figure below. The GPT-2 and GPT-3 largely follow this architecture, with a few modifications and different context size and orders of magnitude regarding the model size.</p> 

<p align="center">
<img src="/assets/img/gpt.png" width="20%" height="20%"> 
</p>
<p align="center"><font size="2">Image adapted from: [4]</font></p>

##### Fine-tuning GPT

<p align="justify">Fine-tuning GPT is the process of adapting the pre-trained model to specific tasks or datasets. After the initial pre-training phase, where GPT learns from a broad corpus of text, fine-tuning involves training the model on a narrower dataset that is specific to the desired application. This additional training helps the model adjust to the nuances and requirements of specific tasks, such as customer service chatbots, medical text analysis, or legal document summarization.</p> 

<p align="justify">During fine-tuning, the model is exposed to labeled examples relevant to the task at hand. This process refines the model’s ability to generate contextually appropriate and accurate responses in a specific domain. Fine-tuning not only enhances the model's performance on specialized tasks but also ensures that it can generate more precise and relevant outputs, making it a versatile tool for various industry applications.</p> 

##### Post-processing in GPT

<p align="justify">Post-processing plays a crucial role in the GPT architecture, ensuring that the generated responses are coherent, contextually appropriate, and aligned with user expectations. After the initial text generation by the model, several post-processing steps are applied to refine and enhance the output, addressing potential issues that may arise during generation.</p> 

<p align="justify">One significant aspect of post-processing involves filtering and moderating the content. The generated text is screened for inappropriate or harmful language, ensuring that the responses are safe and suitable for all users. Additionally, post-processing may include adjusting the tone and style of the responses to better match the intended use case or user preferences.</p>

<p align="justify">Another critical component of post-processing is the handling of coherence and context management. Although the model generates text based on the input provided, it may occasionally produce responses that are off-topic or lack coherence. Post-processing algorithms can detect and correct such issues by re-evaluating the generated text and making necessary adjustments. This ensures that the conversation flows naturally and remains relevant to the user's queries. Furthermore, post-processing can involve the incorporation of external knowledge or context that the model might not have fully captured, enhancing the overall quality and informativeness of the responses.</p> 

<p align="justify">This was just a summary of the GPT architecture, for a more detailed description, please take a look at these references: [4, 9, 14]. We hope that we helped you to grasp a little bit more about the self-attention mechanisms proposed in the Transformer and the GPT model.</p> 

###### **References**

1) [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc)  
2) [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
3) [Stanford CS224N: NLP with Deep Learning](https://www.youtube.com/watch?v=XXtpJxZBa2c)  
4) [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)  
5) [Release Strategies and the Social Impacts of Language Models](https://arxiv.org/pdf/1908.09203)   
6) [Improving language understanding with unsupervised learning](https://openai.com/index/language-unsupervised/)     
7) [GPT-2.5](https://openai.com/index/gpt-2-1-5b-release/)  
8) [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
9) [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)  
10) [Github: OpenAI GPT-2](https://github.com/openai/gpt-2)  
11) [OpenAI - Tokenizer](https://platform.openai.com/tokenizer)  
12) [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)   
13) [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11s)  
14) [GPT-3 Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)  
15) [Neural machine translation](https://arxiv.org/pdf/1409.0473)   
16) [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)   
17) [Understanding Deep Learning, by Simon J.D. Prince (2023), MIT Press](https://udlbook.github.io/udlbook/)  

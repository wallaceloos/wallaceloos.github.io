---
layout: page
title: ChatGPT vs ChatGPT
description: Playing chess with ChatGPT
img: /assets/gifs/black.gif
importance: 1
category: fun
---

[[code]](https://github.com/wallaceloos/Computer_Vision/blob/master/codes/playing_chess_gpt.py)  

<p align="justify"> By the time I am writing this project, OpenAI has recently released a new version of ChatGPT, GPT-4o. Then I decided to check how good this model would be at playing chess. Besides the OpenAI API I also used a python library called 
<a href="https://python-chess.readthedocs.io/en/latest/">python-chess</a>. It is a very handy library for rendering the chessboard, checking illegal movements, etc. </p>


#### Chessboard
  
<p align="justify"> Given a chessboard we can describe the positions of the pieces using the Forsyth-Edwards Notation (FEN) notation. Lowercase letters describe the black pieces, and uppercase letters describe the white pieces, where "p" stands for pawn, "r" for rook, "n" for knight, "b" for bishop, "q" for queen, and "k" for king. The sequence: “'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'” represents the starting position of a chess game, the letter “w” says that its white turns, "KQkq" indicates that  both sides having all castling rights available and the numbers in the end, “0 1”, inform the  completed turns in the game.</p>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <p align="center"> 
        <img src="/assets/img/chessboard.png" width="45%" height="45%"> 
        </p>
    </div>
</div>

#### GPT-4o Player

<p align="justify"> GPT-4o is a large multimodal generative pre-trained transformer designed by OpenAI, that has the impressive capability to perform different tasks, like solving math problems, coding, answering questions, reading information from images, and etc, but can ChatGPT play chess? Let’s find out! The FEN notation, which describes the chess game, was used to let ChatGPT know the position of the pieces over the chessboard. Then,  I created a prompt asking ChatGPT to give me the best move, and I kept doing this until the end of the game (checkmate) or until the game achieved a certain condition, for instance, trying successive illegal moves in a row, or reaching the maximum number of moves. To avoid illegal moves, I checked if the ChatGPT suggestion was valid, if it is not, I ask it again to give me a different answer. 

The movements of the pieces are described using the Universal Chess Interface (UCI), which encodes the start and end coordinates of the piece with 4 characters (letter, digit, letter, digit). An example of UCI notation is: d2d4, which means that the piece is on coordinate d2 and will be moved to  the square d4. In case of promotion, the notation changes to 5 characters, example: e7e8q for a pawn promotion to queen. 
.</p> 

#### Prompts

<p align="justify"> I wrote three different prompts. The first prompt was used every time the fen notation is shown for the first time in the turn for ChatGPT. The second and third prompts were used when ChatGPT tried to make an invalid movement. Two different configurations were used. The first configuration was Prompt 1 & 2, and the second was Prompt 1 & 3. The difference between Prompt 2 and 3 is that Prompt 2 gives ChatGPT the possible moves for that given FEN notation.</p>

```python
Prompt 1: "Given this FEN notation: " + fen + ", give me the best move only in the UCI format. The answer needs to be between ** **. Note, a valid UCI  notation contains a sequence of 4 characters: letter, digit, letter, and digit. However, the notation changes to 5 characters when a piece is promoted. The number ranges from 1 to 8, and the letters can be from 'a' to 'h'."
```

```python
Prompt 2: "This move is invalid:"+ ans_move +". These are the possible moves: "+ str(list_legal_moves) +". Given this FEN notation: " + fen + ", give me the best move only in the UCI format. The answer needs to be between ** **. Note, a valid UCI  notation contains a sequence of 4 characters: letter, digit, letter, and digit. However, the notation changes to 5 characters when a piece is promoted. The number ranges from 1 to 8, and the letters can be from 'a' to 'h'."
```

```python
Prompt 3: "This move is invalid:"+ ans_move + ". Given this FEN notation: " + fen + ", give me the best move only in the UCI format. The answer needs to be between ** **. Note, a valid UCI  notation contains a sequence of 4 characters: letter, digit, letter, and digit. However, the notation changes to 5 characters when a piece is promoted. The number ranges from 1 to 8, and the letters can be from 'a' to 'h'."
```

#### Game

Prompt 1 & 2 (with clues)

<p align="center">
<img src="/assets/gifs/black.gif" width="30%" height="30%">
<img src="/assets/gifs/draw.gif" width="30%" height="30%">
<img src="/assets/gifs/white.gif" width="30%" height="30%">
</p>
<p align="center">
<em>From left to right: Black wins, White wins, and a draw.</em>
</p>

Prompt 1 & 3 (no clues)

<p align="center">
<img src="/assets/gifs/stuck1.gif" width="30%" height="30%">
<img src="/assets/gifs/stuck2.gif" width="30%" height="30%">
<img src="/assets/gifs/stuck3.gif" width="30%" height="30%">
</p>
<p align="center">
<em>ChatGPT cannot suggest any more valid moves after some point.</em>
</p>

<p align="justify"> What I noticed along the game was that at the beginning it was easier for the model to suggest what would be the best move, even when it wasn’t the best one. After some time the number of illegal moves increased a lot, and at some point it looked as if the model was randomly choosing the moves and hallucinating. It wasn’t a total surprise, and I wasn’t expecting ChatGPT to be a grandmaster since it wasn’t trained to play chess (my hypothesis). When no clues about the legal moves are given to ChatGPT, it struggles even more to find a valid move. I believe if I fine-tuning the GPT-4o model using a chess dataset, it would probably improve its performance. Also, It is worth mentioning that this was only my first impression, and I didn’t perform any more sophisticated experiments to evaluate other aspects of the model or to validate my hypothesis.</p>


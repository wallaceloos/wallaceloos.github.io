---
layout: page
title: ML in volleyball
description: ML applied to volleyball
img: /assets/gifs/pipeline_one.gif
importance: 1
category: fun
---

<p align="justify"> Artificial intelligence (A.I.) has been employed in different sports in order to extract information about the players and the game to improve the team’s result. Here I am presenting an example about how A.I. can be applied to volleyball. The pipeline consists of 3 steps. 1) segmenting the volleyball court; 2) creation of a dictionary’s template; and 3) detect and track the players.</p>
  
<p align="center">
<img src="/assets/gifs/pipeline_one.gif" width="100%" height="100%">
</p>

<p align="justify">1) The volleyball court was performed by training a U-net network. Then the segmentation is used to find the most similar template in the template dictionary. This is made in order to find the homography matrix.</p>
  
<p align="center">
<img src="/assets/img/seg.png" width="70%" height="90%">
</p>

<p align="justify">2) The dictionary of templates can be built into two different ways. You can calibrate the camera, and change the extrinsic parameters of the camera, or manually selecting the points over the image and the template, and then moving and rotating the points over the template to create new images.</p>

<p align="center">
<img src="/assets/gifs/templates.gif" width="70%" height="70%">
</p>

<p align="justify">3) To detect the player I trained a YOLO, and to track the players I used the discriminative correlation filter with channel and spatial reliability (CSRT). (Due to the occlusion of the players, YOLO can miss some players)</p>


<p align="center">
<img src="/assets/img/frame.png" width="100%" height="100%">
</p>
  
#### Additional step
  
<p align="justify"> Another step is necessary to make the transition smoother between the frames, the homography matrix refinement step. Sometimes the template does not match entirely with the image, then you can make some adjustment to refine the homography  matrix.</p>
  

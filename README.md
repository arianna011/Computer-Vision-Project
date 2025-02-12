# Optical Music Recognition & Image2Midi app

This is a project for the _Computer Vision_ course at Sapienza a.y. 2024/2025 by Arianna Paolini (1943164), Michele Saraceno (1905065) and Federica Sinisi (1946981).
We based our work on the paper [Using Cell Phone Pictures of Sheet Music To Retrieve MIDI Passages](https://arxiv.org/abs/2004.11724) by Daniel Yang, Thitaree Tanprasert, Teerapat Jenrungrot, Mengyi Shan and TJ Tsai:

<p align="center">
  <img src="\img.png" width="300" height="200">
</p>


we first studied and re-organized their code to implement the **MIDI Retrieval system** (in the directory _Backend\MIDI_Retrieval_System_), which takes in input a query cellphone image of sheet music and returns the temporal segment of the MIDI file of the corresponding piece that best matches the picture. 

We then evaluated it on our laptops and tried to add some functionalities:

- the original system required to get as input not just the query image but also the MIDI file of the corresponding piece, but we thought that it would be useful to implement a **search of the musical piece** on the whole MIDI dataset based on the query image, so that the user does not necessarily need to already have a MIDI file of the piece: we provide the possibility to discover the musical piece through the system, at the cost of increasing the runtime (this functionality can be found in _Backend\main.py_);

- based on our own experience with physical pages of sheet music, we also thought to add the possibility to **retrieve the entire PDF sheet music** of a piece from a picture capturing a single page of it, so that, for example, we can find out which piece an isolated page of sheet music fallen on the floor belongs to (this functionality can be found in _Backend\main.py_).

Finally, we developed a small app, called **Image2Midi**, to leverage the MIDI Retrieval system in a practical way. A user can use this app to take pictures of sheet music, or load them from his gallery, adjust and crop them, hear MIDI passages corresponding to the pictures, downloading the MIDI file of the entire piece or the PDF of the corresponding sheet music, and visualizing all the previously downloaded files.
The implementation of the app can be found in the _App_ directory.

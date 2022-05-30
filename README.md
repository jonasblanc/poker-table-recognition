# Poker playing cards and chips recognition

This repository implements a pipeline to recognize playing cards and chips of a poker table image. We obtain 95% accuracy result on the training set (We used the images on the training set in order to calibrate constants such as: color tresholding for the chips, filters strength (blurring etc), stats for brightness calibration of chips). No machine learning is used in the pipeline so no further "memory" of the training set is created by the pipeline.

<img src="media/example_overlay.png" width="500">

## Table of contents
* [Table extraction](#table-extraction)
* [Table cards extraction](#table-cards-extraction)
* [Player cards extraction](#player-cards-extraction)
* [Card recognition](#card-recognition)
* [Chips recognition](#chips-recognition)
* [Results](#results)

## Table extraction

<img src="https://user-images.githubusercontent.com/44334351/170958205-792b743f-6fbd-4392-9442-9e5007dd00bc.png" width="800">

## Table cards extraction

<img src="https://user-images.githubusercontent.com/44334351/170958230-b3e83f22-5790-4153-83dd-4a4fea434fdd.png" width="800">
<img src="https://user-images.githubusercontent.com/44334351/170958247-24987fad-0f74-4e56-af3c-ca7fbeaa52b1.png" width="800">

## Player cards extraction

<img src="https://user-images.githubusercontent.com/44334351/170958708-b929a6cf-4b6b-4da1-8a1e-be8ba1bc0ac8.png" width="800">

## Card recognition

<img src="https://user-images.githubusercontent.com/44334351/170959309-bb59f914-1ce1-4595-b9e7-9de8ce41ec42.png" width="800">

## Chips recognition

<img src="https://user-images.githubusercontent.com/44334351/170962257-554e1674-574c-40ef-88cb-c7687d952806.png" width="800">
<img src="https://user-images.githubusercontent.com/44334351/170963111-f93cfbbc-a710-4344-a953-7d02fb8b0a4f.png" width="800">

## Results

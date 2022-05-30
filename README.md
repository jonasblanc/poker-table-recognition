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

[Table_extractor](./src/table_extractor.py) implements a pipeline to extract the table from the original image. Steps:  

* Image preprocessing: blurring (median filter to preserve the edges)  
* Edges detection (canny filter)  
* Hough lines  
* Intersection points of lines (that have a cosine similarity < 0.5)   
* Intersection points clustering (with given distance between cluster)  
* Polynomial approximation based on 4 selected cluster centers
* Table extraction

## Table cards extraction

<img src="https://user-images.githubusercontent.com/44334351/170958230-b3e83f22-5790-4153-83dd-4a4fea434fdd.png" width="800">
<img src="https://user-images.githubusercontent.com/44334351/170958247-24987fad-0f74-4e56-af3c-ca7fbeaa52b1.png" width="800">

[Card_extractor](./src/card_extractor.py) implements a pipeline to extract the table cards from the table image. Steps:  

* Crop the table image to select the table cards area  
* Contours extraction: for every hsv channel:
  - K-mean on channel intensity histogram (k=3)
  - Tresholding with every cluster mean
  - Candidate contours extraction
  - Selection of the contours between clusters / channels (smallest variance of perimetre of the 5 largest detected objects)  
* Polynomial approximation on selected contours
* Cards extraction

## Player cards extraction

<img src="https://user-images.githubusercontent.com/44334351/170958708-b929a6cf-4b6b-4da1-8a1e-be8ba1bc0ac8.png" width="800">

[Card_extractor](./src/card_extractor.py) implements a pipeline to extract the players'cards from the table image. Steps:  

* Crop the table image to select one player's cards area   
* Rotation based on player position
* Otsu tresholding
* Contours extraction
* Corners extraction (Polynomial approximation)
* Identify side of the cards (based on corners distance, assume similar card size)
* Estimate position of remaining corners 
* Polynomial approximation based on 4 corners
* Cards extraction (top / bottom label based on the position of the corners toward the corners center)

## Card recognition

<img src="https://user-images.githubusercontent.com/44334351/170959309-bb59f914-1ce1-4595-b9e7-9de8ce41ec42.png" width="800">

[Card_classifier](./src/card_classifier.py) implements a pipeline to classify the extracted card from an image of the card. Steps:  

* Check if cards is face down (= high number of connected components) 
* Crop symbol zone (assume same location on card)
* Extract symbol (largest connected component)
* Identify color (red or black, based on mean symbol color)
* Create a binary centered mask
* Compare it to labeled masks with (intersection / union) score 
* Assign it the label of the mask with highest score
* Similar pipeline for number/letter recognition

## Chips recognition

<img src="https://user-images.githubusercontent.com/44334351/170962257-554e1674-574c-40ef-88cb-c7687d952806.png" width="800">
<img src="https://user-images.githubusercontent.com/44334351/170963111-f93cfbbc-a710-4344-a953-7d02fb8b0a4f.png" width="800">

[Chip_counter](./src/chip_counter.py) implements a pipeline to count the chips by color based on the table image. Steps:  

## Results

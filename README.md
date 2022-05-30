# Poker playing cards and chips recognition

This repository implements a pipeline to recognize playing cards and chips of a poker table image. We obtain 95% accuracy result on the training set (we used the images on the training set in order to calibrate constants such as: color tresholding for the chips, filters strength (blurring etc.), stats for brightness calibration of chips). No machine learning is used in the pipeline so no further "memory" of the training set is created by the pipeline.

<img src="media/example_overlay.png" width="500">

## Table of contents
* [Table extraction](#table-extraction)
* [Table cards extraction](#table-cards-extraction)
* [Player cards extraction](#player-cards-extraction)
* [Card recognition](#card-recognition)
* [Chips recognition](#chips-recognition)
* [Results](#results)
* [Collaborators](#collaborators)

## Table extraction

<img src="https://user-images.githubusercontent.com/44334351/170958205-792b743f-6fbd-4392-9442-9e5007dd00bc.png" width="800">

[Table_extractor](./src/table_extractor.py) implements a pipeline to extract the table from the original image. The steps are:  

* Image preprocessing: blurring (median filter to preserve the edges)  
* Edges detection (canny filter)  
* Hough lines  
* Intersection points of lines (that have a cosine similarity < 0.5)   
* Intersection points clustering (with given distance between clusters)  
* Polynomial approximation based on 4 selected cluster centers
* Table extraction

[Table_extraction](./research%20notebooks/Table_extraction.ipynb) breaks down the pipeline steps by steps. 

## Table cards extraction

<img src="https://user-images.githubusercontent.com/44334351/170958230-b3e83f22-5790-4153-83dd-4a4fea434fdd.png" width="800">
<img src="https://user-images.githubusercontent.com/44334351/170958247-24987fad-0f74-4e56-af3c-ca7fbeaa52b1.png" width="800">

[Card_extractor](./src/card_extractor.py) implements a pipeline to extract the table cards from the table image. The steps are:  

* Crop the table image to select the cards area  
* Contours extraction: for every hsv channel:
  - k-means on channel intensity histogram (k=3)
  - Tresholding with every cluster mean
  - Candidate contours extraction
  - Selection of the contours between clusters / channels (smallest variance of perimeter of the 5 largest detected objects)  
* Polynomial approximation on selected contours
* Cards extraction

## Player cards extraction

<img src="https://user-images.githubusercontent.com/44334351/170958708-b929a6cf-4b6b-4da1-8a1e-be8ba1bc0ac8.png" width="800">

[Card_extractor](./src/card_extractor.py) implements a pipeline to extract the players'cards from the table image. The steps are:  

* Crop the table image to select one player's cards area   
* Rotation based on player position
* Otsu tresholding
* Contours extraction
* Corners extraction (Polynomial approximation)
* Identify side of the cards (based on corners distance, assume similar card size)
* Estimate position of remaining corners 
* Polynomial approximation based on 4 corners
* Cards extraction (top / bottom label based on the position of the corners toward the corners center)

[Player_cards_extraction](./research%20notebooks/Player_cards_extraction.ipynb) breaks down the pipeline steps by steps. 

## Card recognition

<img src="https://user-images.githubusercontent.com/44334351/170959309-bb59f914-1ce1-4595-b9e7-9de8ce41ec42.png" width="800">

[Card_classifier](./src/card_classifier.py) implements a pipeline to classify the extracted card from an image of the card. The steps are:  

* Check if cards are face down (= high number of connected components) 
* Crop symbol zone (assume same location on card)
* Extract symbol (largest connected component)
* Identify color (red or black, based on mean symbol color)
* Create a binary centered mask
* Compare it to labeled masks with (intersection / union) score 
* Assign it to the label of the mask with highest score
A similar pipeline is used for number/letter recognition.

[Card_recognition](./research%20notebooks/Card_recognition.ipynb) can be used to create the groundtruth mask used when classifying cards.

## Chips recognition

<img src="https://user-images.githubusercontent.com/44334351/170962257-554e1674-574c-40ef-88cb-c7687d952806.png" width="800">

[Chip_counter](./src/chip_counter.py) implements a pipeline to count the chips by color based on the table image. The steps are:  

* Hough circles on blurred gray image
* For each circle, select color with largest number of pixel in the circle:
  * Brightness equalisation and median filtering
  * For each color: Binary HSV thresholding
  * Intersection of binary masks between circle and color masks

[Chips_preprocessing](./research%20notebooks/Chips_preprocessing.ipynb) can be used to compute the color cluster and to merge chips images into one which is convenient for finding color thresholds.

In order to find the HSV threshold for the color of the chips (red, blue, green, black, white), we implemented an interface to be able to see the effect of the thresholding directly on most of the chips. [Chips_colors_thresholds](./research%20notebooks/Chips_colors_thresholds.ipynb) can be used to launch the interface.

<img src="https://user-images.githubusercontent.com/44334351/170963111-f93cfbbc-a710-4344-a953-7d02fb8b0a4f.png" width="800">


## Results

The training set is composed of 28 images of poker hands with different illuminations​

* 94.7 % of accuracy on the number of the cards
* 92.5 % of accuracy  on the suits of the cards
* 99.0 % of accuracy on the number of chips for each color

## Collaborators

@esteegdd - Estee Grandidier  
@Squalene - Antoine Masanet  
@jonasblanc - Jonas Blanc  

# Multimodal Fungi Challenge 2025 - Mushrooms, Metadata and Machine Learning!
Welcome to Multimodal Fungi Challenge 2025, the summer school challenge where algorithms meet mushrooms, and machine learning meets metadata! Strap on your data scientist hat (preferably fungus-shaped) because this challenge is going to be an exploration of Danish fungi like never before.

This challenge is made in conjunction with the [Ph. D. Summer school on multi-modal learning](https://multi-modal.compute.dtu.dk/).

This summer school challenge revolves around multi-modal learning, thus combining multiple types of input data to create better and smarter classification models. 
Your mission, should you choose to accept it, is to build a model that can correctly classify images of fungi using both the images and a treasure trove of fascinating metadata.

You are tasked with answering questions like:
- Is the fungus photogenic or just "photo-fungic"?
- Does its habitat have the winning vibes of fungi?
- Can latitude and longitude reveal the secret of mushroom mastery?
- Does seasonality hold the key to its identity?
- Can appearance be deceiving?

Here’s what you need to conquer during the challenge:

**Score That F1!** Develop a model to predict the right label and achieve sky-high F1 scores for fungus classification.

**Mix It Up!** Explore different multimodal learning strategies, figuring out clever ways to leverage metadata.

**Metadata Treasure Hunt!** Unearth what metadata helps the most in boosting classification performance.

Let’s face it, this challenge isn’t just about cross-validation scores. It’s your chance to engage with multimodal data, experiment, and create fungi-classifying algorithms that would make both a biologist and a data scientist proud.
So get ready to build clever pipelines, negotiate with metadata, and battle it out with other teams in the quest to become the Master of Mushrooms! Let’s bring both rigor and innovation to the Multimodal Fungi Challenge 2025. 🍄

## Why fungi classification, and why multi-modal learning?
Mushrooms, while often delicious, can toe the line between harmless delicacies and unexpected dangers. Accidental poisonings around the world highlight the need for reliable identification methods. In one chilling case in Australia, a woman is on trial for serving toxic fungi in a Beef Wellington and killing three people!
The stakes are clear: what if an AI could have flagged those mushrooms before tragedy struck? Accurate fungi classification, powered by advanced algorithms, could save lives and prevent disasters where human judgment falls short.

Photos alone often don't tell the full story of a mushroom’s identity. Enter metadata, the contextual clues that surround the image:

- **Time (Seasonality):** Fungi are seasonal performers, sprouting in specific months. Metadata about the date offers critical cues into their behavior.
- **Location (Latitude and Longitude):** Where a fungus grows can often narrow down its identity, providing geographical hints specific to Denmark’s ecosystems.
- **Habitat:** A mushroom growing in dense, moist forests isn’t quite the same as one flourishing in open meadows—habitat descriptions matter!
- **Substrate:** What it grows on (tree stumps, soil, sand) is sometimes the biggest hint of all.

The real power comes from combining these two modalities—visual features and contextual metadata. While a photo may show a glorious mushroom cap, it’s the metadata that reveals the subtle cues hiding just beneath the surface: is it poisonous? Rare? Seasonal? Found only in one treasured corner of Denmark?

## Data

There are 183 different types of fungi found in Denmark in this challenge. That means that the label is an integer in the range [0, 182].

The metadata consists of:
- The `eventDate` when the photo was taken (day, month, year)
- The `Latitude` of the position of the fungi
- The `Longitude` of the position of the fungi
- The `Habitat` where the fungi was found (textual description)
- The `Substrate` that the fungi was growing on (textual description)

There are three sets of data in the challenge:
- **Training:** Where the photos are named `fungi_trainXXXXXX.jpg`, where XXXXXXX is a zero-padded integer. For the training set, the label (taxonID_index) is given and you can buy metadata (a subset of metadata is available at the start of the challenge).
- **Testing:** Where the photos are named `fungi_testXXXXXX.jpg`, where XXXXXXX is a zero-padded integer. This set is used to give team score during the challenge. All metadata is available and the true label should be predicted.
- **Final:** Where the photos are named `fungi_finalXXXXXX.jpg`, where XXXXXXX is a zero-padded integer. This set is used to compute the final challenge score. All metadata is available and true label should be predicted.

## Preparing for the challenge
- Please download the [challenge image data](http://fungi.compute.dtu.dk:8080/downloads/FungiImages.zip) before the challenge (13 GB of data).
- Place the data on a cluster that you will have access to during the challenge (because fungi deserve room to grow... in processing power).

## Challenge web site

The main challenge website can be found [here](http://fungi.compute.dtu.dk:8080)

### Team logins

At the summer school, you will get a team name and its password, so you can login to the challenge site.

### Team Dashboard

At the team dashboard on the challenge website on you can:
- Buy more metadata for the training set
- Download your current metadata
- Upload predictions for the test and the final set

### Getting your metadata

To get your current metadata you should:
- Press the **Generate your data** link on the Dashboard page.
- Press the **Get your data** link on the Dashboard page.

## Supplied Python Scripts
In the 'src' folder you will find a few scripts to help you get started on your fungi adventure.
- `create_shoppinglist.py`: Will help you prepare your metadata-shoppinglist and save it as an .csv file
- `fungi_network.py`: Will train a simple EfficientNet for classification of fungi images and create predictions saved as a .csv file
- `random_fungi_predictions.py`: Randomly assigns labels to the test set

## Getting Started
This should get you started in the challenge:

- Download the image data
- Login with your team name and password
- Prepare your data
- Get your data
- Adapt and run the `random_fungi_predictions.py` example script
- Use the basic image classification pipeline in `fungi_network.py` and train a network on the imaging data
- Upload the predictions
- Investigate the available metadata in the training set, and cultivate the ultimate strategy for your fungi-fueled team! 🍄
- **Have fun(gi)!**

### Buying more metadata

You buy metadata by preparing a CSV file and upload it in the Dashboard page. 

The meta data costs:
- `eventDate` : 2 credits
- `Latitude` : 1 credit
- `Longitude` : 1 credit
- `Habitat` : 2 credits
- `Substrate` : 2 credits

The entries/rows in the CSV file are **image name**, **metadata type**. You can use the script in `create_shoppinglist.py` found in the `src` folder to generate the .csv file in the correct format.

For example:

```
fungi_train000000.jpg,Habitat
fungi_train000002.jpg,Latitude
fungi_train000018.jpg,Longitude
fungi_train000019.jpg,Substrate
fungi_train000018.jpg,eventDate
``` 

This example will spend eight credits to buy the specified metadata for the specified photos.

When you have bought metadata, there should be a summary message at the bottom of the Dashboard page.

When you have bought metadata, you should *generate* and *get* your data.

### Send more money - how much do I have?

Each team starts with 20.000 credits. During the challenge, this amount increases with 10.000 every 12 hours.

### Uploading predictions

You upload predictions by preparing a CSV file, where each row contains the **image name** and the **label**

**NOTE:** The first line of the CSV file should be a short textual description of the prediction method.

Example:
```
random_fungi_predictions
fungi_test000000.jpg,99
fungi_test000001.jpg,2
fungi_test000002.jpg,107
fungi_test000003.jpg,173
...
```

When you have uploaded a prediction file, there should be a summary message at the bottom of the Dashboard page.

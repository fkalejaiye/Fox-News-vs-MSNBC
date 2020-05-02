# Proposals for Capstone 2

## Big Cats
>This project will focus on creating a model that can accurately predict the species of a big cat given a picture. All 7 species of big cats will be considered.

>My approach to this will be to get a large collection of images of different species from google via web scraping. I will then transform them into a format in which I can use them in python. I will then use various regression types to try and fit my data and come up with a model.

>Users will be able to interact with this via a flask app that I will create. The user will be able to submit a picture of a big cat and then receive a prediction of what type of big cat it is, along with some information about the particular species.

## House Prices
> This project will attempt to create a model that given a zip code, square footage, and other features, can accurately predict the selling price of a house in the Austin Texas area.

>My aproach will be to scrape the website realtor.com for homes within the austin texas area. I will compile a data set of each home with it's corresponding features and use this to fit a model. 

>Users will interact with this via a flask app that I will create in which a user can enter certain information and receive a predicted price.

## Housing Type
>At my previous job in insurance i found that the architectural type of a home was an integral part in calculating its insurance premium. However as someone who was not familiar with this subject it was often difficult to accurately classify. To solve this problem I intend to create a model that given a picture of a home can accurately classify the architectural type as one of the 10 most common types.

>My approach would be to scrape data from google by searching google images for pictures of homes for each type and then transforming the data into a workable format. I would then fit the data to a model. 

>Users will be able to interact with my model via a flask app that I will create. They will be able to submit a picture of a home and it will give them the architectural type that the home most closely resembles. As well as a description of the architectural type with a breif history.
# PIR

## Twitch Game Data Marketplace Simulation

This project simulates a marketplace for Twitch game data using a logistic classifier and linear programming. The simulation aims to maximize the market surplus by matching bids from buyers (brands) and sellers (games) based on their game-related data.

### Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Marketplace Simulation](#marketplace-simulation)
- [Running the Simulation](#running-the-simulation)
- [Functions and Classes](#functions-and-classes)
  - [Data Loading](#data-loading)
  - [Marketplace Simulation](#marketplace-simulation-1)
  - [Logistic Classifier](#logistic-classifier)
  - [Optimization](#optimization)
  - [Brands Class](#brands-class)
  - [Main Function](#main-function)
- [Plots](#plots)
- [Future Improvements](#future-improvements)

### Introduction
This project is designed to simulate a marketplace where Twitch game data is traded. The simulation uses logistic regression to predict the maximum bid values (`bmax`) for buyers and linear programming to determine the winning bids in the marketplace, thereby maximizing the market surplus.

### Dependencies
Ensure you have the following Python libraries installed:
- `pulp`
- `numpy`
- `csv`
- `collections`
- `sklearn`
- `matplotlib`

You can install these libraries using pip:
```bash
pip install pulp numpy scikit-learn matplotlib
```

### Data Preparation
The input data file is a CSV file named `Twitch_game_data.csv` containing Twitch game data with the following columns:
- `Game`
- `Year`
- `Month`
- `Hours_watched`
- `Hours_streamed`
- `Peak_viewers`
- `Peak_channels`
- `Streamers`
- `Avg_viewers`

The data is loaded into a nested dictionary structure to facilitate easy access and manipulation.

### Marketplace Simulation
The marketplace simulation involves two main parts: the seller side and the buyer side. Sellers offer bids based on game data, and buyers use a logistic classifier to determine their maximum bid values (`bmax`). The bids are then optimized using linear programming to maximize the market surplus.

### Running the Simulation
To run the simulation, simply execute the script:
```bash
python marketplace.py
```

### Functions and Classes

#### Data Loading
- **`defaultdict_to_dict(d)`**: Recursively converts a `defaultdict` to a regular dictionary. Loads data from the CSV file into a nested dictionary, filters out games with fewer than 4 entries, and converts the defaultdict to a regular dictionary.

#### Marketplace Simulation
- **`simulate_marketplace(nb_acheteurs, donnees_dict)`**: Simulates the marketplace with a specified number of buyers for all the games. This function includes:
  - **Seller Side**: Generates bids based on game data.
  - **Buyer Side**: Uses a logistic classifier to determine `bmax` for buyers and generates their bids.
  - **Optimization**: Uses linear programming to maximize the market surplus by matching bids.

#### Logistic Classifier
- **`classifier(jeu)`**: Trains a logistic regression model to predict the maximum bid (`bmax`) for a given game. This involves:
  - Scaling the data.
  - Splitting the data into training and testing sets.
  - Training the logistic regression model.
  - Calculating the prediction error and determining `bmax`.

#### Optimization
- **`optimization(new_bids)`**: Optimizes the bids using linear programming to maximize the market surplus. This involves:
  - Setting up the linear programming problem.
  - Adding constraints to ensure bids are feasible.
  - Solving the problem and identifying the winning bids.

#### Brands Class
- **`Brands` Class**: Represents buyers (brands) in the marketplace. Methods include:
  - **`generate_bid()`**: Generates a bid based on the logistic classifier.
  - **`set_q(all_q)`**: Modifies the bid vector (`q`) based on the game data.

#### Main Function
- **`main()`**: Runs the simulation for different numbers of buyers and plots the results.

### Plots
The main function generates a plot showing the market surplus versus the number of agents. This helps visualize how the market surplus changes with different numbers of buyers.

### Future Improvements
- Improve the logistic regression model for better accuracy.
- Experiment with different machine learning models.
- Optimize the selection of the threshold in the classifier function.
- Enhance the visualization and analysis of results.
- Display other plots showing execution time versus the number of agents or the surplus versus other parameters.

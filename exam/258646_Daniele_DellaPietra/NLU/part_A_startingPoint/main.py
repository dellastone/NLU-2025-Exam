# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
"""
## Part 2.A
As for the first part of the project (LM), you have to apply these two modifications incrementally. Also in this case you may have to play with the hyperparameters and optimizers to improve the performance. 

Modify the baseline architecture Model IAS by:
- Adding bidirectionality
- Adding dropout layer

**Intent classification**: accuracy <br>
**Slot filling**: F1 score with conll

***Dataset to use: ATIS***

"""
# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    args = get_arguments()
    
    if args.train:
        start_training(args)
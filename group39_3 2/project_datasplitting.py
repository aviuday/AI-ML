import splitfolders

splitfolders.ratio("Data/Celebs_15_detected", # The location of dataset
                   output="Data/Celebs_15_Training_and_Validation", # The output location
                   seed=42, # The number of seed
                   ratio=(.85, .15, 0), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
)